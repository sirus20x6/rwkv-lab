// Package authoritygate enforces the dashboard's observation-only process
// boundary at source level. Training lifecycle authority belongs to TrainVM;
// server and alert projections must not grow a second process-control path.
package authoritygate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

type authorityViolation struct {
	position token.Position
	rule     string
}

func TestServerAndAlertsHaveNoDirectTrainingProcessAuthority(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate authority gate source")
	}
	internalRoot := filepath.Dir(filepath.Dir(thisFile))
	fileSet := token.NewFileSet()
	var violations []authorityViolation
	for _, packageName := range []string{"server", "alerts"} {
		root := filepath.Join(internalRoot, packageName)
		err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() || filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			parsed, err := parser.ParseFile(fileSet, path, nil, 0)
			if err != nil {
				return err
			}
			imports := importedPackageNames(parsed)
			ast.Inspect(parsed, func(node ast.Node) bool {
				call, callOK := node.(*ast.CallExpr)
				if !callOK {
					return true
				}
				selector, selectorOK := call.Fun.(*ast.SelectorExpr)
				if !selectorOK {
					return true
				}
				if selector.Sel.Name == "Start" {
					violations = append(violations, authorityViolation{
						position: fileSet.Position(call.Pos()),
						rule:     "direct process Start",
					})
					return true
				}
				receiver, receiverOK := selector.X.(*ast.Ident)
				if !receiverOK {
					return true
				}
				importPath := imports[receiver.Name]
				switch {
				case importPath == "os" && selector.Sel.Name == "StartProcess":
					violations = append(violations, authorityViolation{
						position: fileSet.Position(call.Pos()),
						rule:     "os.StartProcess",
					})
				case importPath == "syscall" && selector.Sel.Name == "Kill" && !isSignalZeroProbe(call):
					violations = append(violations, authorityViolation{
						position: fileSet.Position(call.Pos()),
						rule:     "syscall.Kill with a nonzero signal",
					})
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatalf("scan %s source: %v", packageName, err)
		}
	}
	for _, violation := range violations {
		relative, err := filepath.Rel(internalRoot, violation.position.Filename)
		if err != nil {
			relative = violation.position.Filename
		}
		t.Errorf("%s:%d: %s violates the observation-only dashboard boundary",
			relative, violation.position.Line, violation.rule)
	}
}

func importedPackageNames(file *ast.File) map[string]string {
	result := make(map[string]string, len(file.Imports))
	for _, declaration := range file.Imports {
		path, err := strconv.Unquote(declaration.Path.Value)
		if err != nil || declaration.Name != nil && declaration.Name.Name == "_" {
			continue
		}
		name := filepath.Base(path)
		if declaration.Name != nil {
			name = declaration.Name.Name
		}
		result[name] = path
	}
	return result
}

// syscall.Kill(pid, 0) is an observation-only existence probe. It neither
// signals nor changes the target process and is the sole process-control-shaped
// exception admitted by this gate.
func isSignalZeroProbe(call *ast.CallExpr) bool {
	if len(call.Args) != 2 {
		return false
	}
	literal, ok := call.Args[1].(*ast.BasicLit)
	if !ok || literal.Kind != token.INT {
		return false
	}
	value, err := strconv.ParseInt(literal.Value, 0, 64)
	return err == nil && value == 0
}
