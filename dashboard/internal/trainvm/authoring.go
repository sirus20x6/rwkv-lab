package trainvm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
)

// Authoring exposes immutable schema/example documents and delegates semantic
// compilation to the native TrainVM binary. It never writes or launches a run.
type Authoring struct {
	BinaryPath  string
	SchemaPath  string
	ExamplePath string
}

func (a *Authoring) Schema() (json.RawMessage, error) {
	return readJSONDocument(a.SchemaPath, "TrainVM schema")
}

func (a *Authoring) Example() (json.RawMessage, error) {
	return readJSONDocument(a.ExamplePath, "TrainVM example")
}

func readJSONDocument(path, label string) (json.RawMessage, error) {
	if path == "" {
		return nil, fmt.Errorf("%s path is not configured", label)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	if !json.Valid(data) {
		return nil, fmt.Errorf("%s is not valid JSON", label)
	}
	return json.RawMessage(data), nil
}

func (a *Authoring) Compile(ctx context.Context, document []byte) (json.RawMessage, error) {
	if a.BinaryPath == "" {
		return nil, fmt.Errorf("TrainVM compiler path is not configured")
	}
	if !json.Valid(document) {
		return nil, fmt.Errorf("experiment draft is not valid JSON")
	}
	command := exec.CommandContext(ctx, a.BinaryPath, "compile")
	command.Stdin = bytes.NewReader(document)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	if err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) || exitError.ExitCode() != 2 {
			if contextError := ctx.Err(); contextError != nil {
				return nil, fmt.Errorf("native TrainVM compile timed out: %w", contextError)
			}
			return nil, fmt.Errorf("native TrainVM compile failed: %v: %s", err, stderr.String())
		}
	}
	if !json.Valid(stdout.Bytes()) {
		return nil, fmt.Errorf("native TrainVM compiler returned invalid JSON: %s", stderr.String())
	}
	return json.RawMessage(bytes.Clone(stdout.Bytes())), nil
}
