package trainvm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuthoringReadsDocumentsAndRunsNativeCompiler(t *testing.T) {
	directory := t.TempDir()
	schema := filepath.Join(directory, "schema.json")
	example := filepath.Join(directory, "example.json")
	binary := filepath.Join(directory, "trainvm")
	if err := os.WriteFile(schema, []byte(`{"title":"schema"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(example, []byte(`{"kind":"Experiment"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{\"valid\":true,\"plan_hash\":\"abc\"}'\n"
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	authoring := &Authoring{BinaryPath: binary, SchemaPath: schema, ExamplePath: example}
	if got, err := authoring.Schema(); err != nil || string(got) != `{"title":"schema"}` {
		t.Fatalf("schema=%s err=%v", got, err)
	}
	if got, err := authoring.Example(); err != nil || string(got) != `{"kind":"Experiment"}` {
		t.Fatalf("example=%s err=%v", got, err)
	}
	result, err := authoring.Compile(context.Background(), []byte(`{"kind":"Experiment"}`))
	if err != nil || !strings.Contains(string(result), `"plan_hash":"abc"`) {
		t.Fatalf("compile=%s err=%v", result, err)
	}
}

func TestAuthoringAcceptsStructuredNativeDiagnostics(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "trainvm")
	script := "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{\"valid\":false,\"diagnostics\":[{\"code\":\"field.required\"}]}'\nexit 2\n"
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	authoring := &Authoring{BinaryPath: binary}
	result, err := authoring.Compile(context.Background(), []byte(`{}`))
	if err != nil || !strings.Contains(string(result), `"valid":false`) {
		t.Fatalf("compile=%s err=%v", result, err)
	}
	if _, err := authoring.Compile(context.Background(), []byte(`{`)); err == nil {
		t.Fatal("invalid draft JSON was accepted")
	}
}
