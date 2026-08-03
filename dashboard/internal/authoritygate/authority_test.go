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

type auditedExecSite struct {
	relativePath string
	function     string
	constructor  string
}

func (site auditedExecSite) key() string {
	return filepath.Join(site.relativePath) + ":" + site.function + ":" + site.constructor
}

func TestServerAndAlertsHaveNoDirectTrainingProcessAuthority(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate authority gate source")
	}
	internalRoot := filepath.Dir(filepath.Dir(thisFile))
	fileSet := token.NewFileSet()
	var violations []authorityViolation
	auditedExecSites := map[string]int{
		(auditedExecSite{"server/arch.go", "handleArchitecture", "CommandContext"}).key():         0,
		(auditedExecSite{"server/posttraining.go", "handleInspectPosttraining", "Command"}).key(): 0,
	}
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
				if selector.Sel.Name == "Signal" {
					violations = append(violations, authorityViolation{
						position: fileSet.Position(call.Pos()),
						rule:     "direct process Signal",
					})
					return true
				}
				receiver, receiverOK := selector.X.(*ast.Ident)
				if !receiverOK {
					return true
				}
				importPath := imports[receiver.Name]
				switch {
				case importPath == "os/exec" &&
					(selector.Sel.Name == "Command" || selector.Sel.Name == "CommandContext"):
					relative, relativeErr := filepath.Rel(internalRoot, path)
					if relativeErr != nil {
						relative = path
					}
					site := auditedExecSite{
						relativePath: filepath.ToSlash(relative),
						function:     enclosingFunction(parsed, call.Pos()),
						constructor:  selector.Sel.Name,
					}
					key := site.key()
					if _, allowed := auditedExecSites[key]; !allowed ||
						!hasAuditedReadOnlyShape(site, call) {
						violations = append(violations, authorityViolation{
							position: fileSet.Position(call.Pos()),
							rule:     "unaudited dashboard subprocess",
						})
					} else {
						auditedExecSites[key]++
					}
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
	for site, count := range auditedExecSites {
		if count != 1 {
			t.Errorf("audited read-only subprocess %s occurred %d times; want exactly one", site, count)
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

func enclosingFunction(file *ast.File, position token.Pos) string {
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if ok && function.Pos() <= position && position <= function.End() {
			return function.Name.Name
		}
	}
	return ""
}

func hasAuditedReadOnlyShape(site auditedExecSite, call *ast.CallExpr) bool {
	switch site.key() {
	case (auditedExecSite{"server/arch.go", "handleArchitecture", "CommandContext"}).key():
		return len(call.Args) == 4 && identifier(call.Args[0]) == "ctx" &&
			identifier(call.Args[1]) == "py" && identifier(call.Args[2]) == "wrapper" &&
			identifier(call.Args[3]) == "runDir"
	case (auditedExecSite{"server/posttraining.go", "handleInspectPosttraining", "Command"}).key():
		return len(call.Args) == 7 && stringLiteral(call.Args[1]) == "-m" &&
			stringLiteral(call.Args[2]) == "rwkv_lab.posttrain_data" &&
			identifier(call.Args[3]) == "path" && stringLiteral(call.Args[4]) == "--limit" &&
			stringLiteral(call.Args[5]) == "3" && stringLiteral(call.Args[6]) == "--json"
	default:
		return false
	}
}

func identifier(expression ast.Expr) string {
	value, ok := expression.(*ast.Ident)
	if !ok {
		return ""
	}
	return value.Name
}

func stringLiteral(expression ast.Expr) string {
	literal, ok := expression.(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return ""
	}
	value, err := strconv.Unquote(literal.Value)
	if err != nil {
		return ""
	}
	return value
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
