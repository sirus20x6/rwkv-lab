package db

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// PR #223 gave this package a one-token writer semaphore and routed every
// durable write through it, but left *sql.DB embedded in DB. The token
// therefore held only by convention: any package with a *db.DB could write
// d.DB.Exec(...) or d.DB.Begin() and contend with ingest exactly as before —
// the same "asserted in a doc comment, enforced by nothing" shape as the defect
// the token was added to fix.
//
// The primary fix is the type, not this test: the handle is now an unexported,
// non-embedded field, so outside this package the bypass does not compile at
// all. The compiler cannot police the one place that still reaches the raw
// handle, which is this package itself, and these two tests cover exactly that
// residue.
//
// TestDBHandleStaysUnexportedAndUnembedded guards the type change from being
// undone; TestRawSQLHandleIsReachedOnlyByTheWriterHelpers guards the in-package
// call sites. Neither is a substitute for the other: re-embedding the handle
// restores the bypass for every other package without adding one call site
// here, and a new in-package writer bypasses the token without touching the
// struct.

// writeHelpers may reach the raw handle with any method: they are the
// functions that hold the writer token across the call.
var writeHelpers = map[string]bool{
	"Exec":       true, // holds the token for one statement
	"TryExec":    true, // takes the token or skips the write entirely
	"beginWrite": true, // holds the token for the transaction's lifetime
}

// readForwarders may reach the raw handle only with methods in readMethods.
// They exist because the handle is no longer embedded, so the pooled read
// surface has to be re-offered explicitly — that explicitness is the point:
// promotion was all-or-nothing and carried Exec, Begin, BeginTx and Prepare
// with it.
var readForwarders = map[string]bool{
	"Query":           true,
	"QueryContext":    true,
	"QueryRow":        true,
	"QueryRowContext": true,
	"Close":           true,
	"Ping":            true,
	"Stats":           true,
}

// readMethods never take the writer token, and must not: WAL exists so the
// live UI keeps being served underneath a multi-minute ingest transaction.
// Serializing reads behind the token would reintroduce the stall from the
// other side.
var readMethods = map[string]bool{
	"Query": true, "QueryContext": true,
	"QueryRow": true, "QueryRowContext": true,
	"Ping": true, "PingContext": true,
	"Close": true, "Stats": true,
	"SetMaxOpenConns": true, "SetMaxIdleConns": true,
	"SetConnMaxLifetime": true, "SetConnMaxIdleTime": true,
}

// parsePackageSources parses the non-test sources of this package. Test files
// are excluded deliberately: the defect is a production-concurrency one, and a
// _test.go file is not linked into the binary that runs beside a live ingest.
func parsePackageSources(t *testing.T) (*token.FileSet, []*ast.File) {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package directory: %v", err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(".", name), nil, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		files = append(files, parsed)
	}
	if len(files) == 0 {
		t.Fatal("parsed no package sources; the check would pass vacuously")
	}
	return fset, files
}

// rawHandleField returns the name of DB's *sql.DB field, having asserted there
// is exactly one and that it is neither embedded nor exported. Deriving the
// name rather than hardcoding it is what makes the second test survive a
// rename, and what makes a *second* raw handle — a bypass by addition rather
// than by call site — fail here.
func rawHandleField(t *testing.T, files []*ast.File) string {
	t.Helper()
	var found *ast.StructType
	for _, file := range files {
		ast.Inspect(file, func(n ast.Node) bool {
			spec, ok := n.(*ast.TypeSpec)
			if !ok || spec.Name.Name != "DB" {
				return true
			}
			if structType, ok := spec.Type.(*ast.StructType); ok {
				found = structType
			}
			return true
		})
	}
	if found == nil {
		t.Fatal("no type DB struct in package db")
	}

	var handles []string
	for _, field := range found.Fields.List {
		star, ok := field.Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		selector, ok := star.X.(*ast.SelectorExpr)
		if !ok {
			continue
		}
		pkg, ok := selector.X.(*ast.Ident)
		if !ok || pkg.Name != "sql" || selector.Sel.Name != "DB" {
			continue
		}
		if len(field.Names) == 0 {
			t.Fatalf("DB embeds *sql.DB: every package holding a *db.DB can then " +
				"call d.DB.Exec or d.DB.Begin and contend with ingest, which is the " +
				"bypass the unexported field exists to make inexpressible")
		}
		for _, name := range field.Names {
			if ast.IsExported(name.Name) {
				t.Fatalf("DB.%s is an exported *sql.DB handle: other packages can write "+
					"through it without taking the writer token", name.Name)
			}
			handles = append(handles, name.Name)
		}
	}
	if len(handles) != 1 {
		t.Fatalf("DB holds %d *sql.DB fields (%v); expected exactly one, because the "+
			"check below polices reaches to that one handle", len(handles), handles)
	}
	return handles[0]
}

func TestDBHandleStaysUnexportedAndUnembedded(t *testing.T) {
	_, files := parsePackageSources(t)
	if handle := rawHandleField(t, files); handle == "" {
		t.Fatal("empty handle name")
	}
}

// TestRawSQLHandleIsReachedOnlyByTheWriterHelpers fails on any reach to the raw
// pool from a function that is not one of the writer helpers, and on any
// non-read method reached from a read forwarder.
//
// The predicate is "reaches the handle at all", not "calls Exec on it",
// because assigning the handle to a local (p := d.pool) or returning it defeats
// a method-name rule entirely while looking innocuous. A read is still allowed
// through the read forwarders — pooled readers are the whole reason the pool
// keeps four connections, so a rule that banned every raw reach would be
// unimplementable rather than strict.
//
// A new function that legitimately needs the handle fails here until it is
// added to one of the two lists above. That friction is the mechanism: it costs
// one line and turns "I remembered the token" into a recorded decision.
func TestRawSQLHandleIsReachedOnlyByTheWriterHelpers(t *testing.T) {
	fset, files := parsePackageSources(t)
	handle := rawHandleField(t, files)

	var violations []string
	for _, file := range files {
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			isWriteHelper := writeHelpers[fn.Name.Name]
			isReadForwarder := readForwarders[fn.Name.Name]

			ast.Inspect(fn.Body, func(n ast.Node) bool {
				selector, ok := n.(*ast.SelectorExpr)
				if !ok || selector.Sel.Name != handle {
					return true
				}
				where := fset.Position(selector.Pos())
				switch {
				case isWriteHelper:
					// Holds the token; any method is legitimate.
				case isReadForwarder:
					// Only a read may be forwarded straight to the pool.
					method, isMethodCall := reachedMethod(n, fn.Body)
					if !isMethodCall {
						violations = append(violations, where.String()+": "+fn.Name.Name+
							" is a read forwarder but does something other than call a method on the pool")
						break
					}
					if !readMethods[method] {
						violations = append(violations, where.String()+": read forwarder "+
							fn.Name.Name+" reaches the pool with "+method+
							", which is not a read; a durable write must go through Exec, TryExec or beginWrite")
					}
				default:
					method, isMethodCall := reachedMethod(n, fn.Body)
					detail := "takes a reference to the raw pool"
					if isMethodCall {
						detail = "calls " + method + " on the raw pool"
					}
					violations = append(violations, where.String()+": "+fn.Name.Name+" "+detail+
						", bypassing the writer token; use Exec, TryExec or beginWrite, or add "+
						fn.Name.Name+" to writeHelpers/readForwarders with a reason")
				}
				return true
			})
		}
	}

	if len(violations) > 0 {
		sort.Strings(violations)
		t.Fatalf("raw *sql.DB handle reached outside the writer helpers:\n  %s",
			strings.Join(violations, "\n  "))
	}
}

// reachedMethod reports the method selected on the handle, if the reach is of
// the form x.<handle>.Method. Anything else — a bare reference, an assignment,
// a return — is reported by the caller as a reference rather than a call, which
// is the case a method-name allowlist would silently miss.
func reachedMethod(handleSelector ast.Node, body *ast.BlockStmt) (string, bool) {
	var method string
	ast.Inspect(body, func(n ast.Node) bool {
		outer, ok := n.(*ast.SelectorExpr)
		if !ok || outer.X != handleSelector {
			return true
		}
		method = outer.Sel.Name
		return false
	})
	return method, method != ""
}
