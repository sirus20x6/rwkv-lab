package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"

	trainvmstore "trainboard/internal/trainvm"
)

const trainVMDescriptorLimit = 1 << 20

type trainVMDescriptorSpec struct {
	provider   string
	version    string
	apiVersion string
	listField  string
	validate   func([]any) error
}

var (
	trainVMTrainingComponentsDescriptor = trainVMDescriptorSpec{
		provider: "trainvm.training-components", version: "1.0.0",
		apiVersion: "trainvm.training-components/v1", listField: "components",
		validate: validateTrainingComponentDescriptors,
	}
	trainVMOperationsDescriptor = trainVMDescriptorSpec{
		provider: "trainvm.operations", version: "1.0.0",
		apiVersion: "trainvm.operations/v1", listField: "operations",
		validate: validateOperationDescriptors,
	}
)

func (s *Server) handleTrainVMTrainingComponents(w http.ResponseWriter, r *http.Request) {
	s.handleTrainVMDescriptor(w, r, trainVMTrainingComponentsDescriptor)
}

func (s *Server) handleTrainVMOperations(w http.ResponseWriter, r *http.Request) {
	s.handleTrainVMDescriptor(w, r, trainVMOperationsDescriptor)
}

func (s *Server) handleTrainVMDescriptor(w http.ResponseWriter, r *http.Request,
	spec trainVMDescriptorSpec) {
	w.Header().Set("Cache-Control", "no-store")
	if s.commander == nil {
		http.Error(w, "TrainVM authority is not configured", http.StatusServiceUnavailable)
		return
	}
	result, err := s.commander.GetDescriptor(r.Context(), trainvmstore.DescriptorRequest{
		Provider: spec.provider, Version: spec.version,
	})
	if err != nil {
		writeTrainVMAuthorityError(w, err)
		return
	}
	document, err := validateTrainVMDescriptor(result, spec)
	if err != nil {
		http.Error(w, "native authority returned an invalid descriptor document", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"schema_hash": result.SchemaHash,
		"schema":      document,
	})
}

func validateTrainVMDescriptor(result trainvmstore.DescriptorResult,
	spec trainVMDescriptorSpec) (map[string]any, error) {
	raw := []byte(result.SchemaJSON)
	if len(raw) == 0 || len(raw) > trainVMDescriptorLimit {
		return nil, fmt.Errorf("descriptor size is outside the authority bound")
	}
	decoded, err := decodeCanonicalJSONObject(raw)
	if err != nil {
		return nil, err
	}
	if !exactObjectKeys(decoded, []string{"api_version", spec.listField}, nil) {
		return nil, fmt.Errorf("descriptor top-level envelope is not exact")
	}
	apiVersion, ok := decoded["api_version"].(string)
	if !ok || apiVersion != spec.apiVersion {
		return nil, fmt.Errorf("descriptor api_version is not the requested contract")
	}
	items, ok := decoded[spec.listField].([]any)
	if !ok {
		return nil, fmt.Errorf("descriptor list has the wrong type")
	}
	if spec.validate != nil {
		if err := spec.validate(items); err != nil {
			return nil, err
		}
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(raw))
	if result.SchemaHash != digest {
		return nil, fmt.Errorf("descriptor identity does not match its canonical bytes")
	}
	return decoded, nil
}

// decodeCanonicalJSONObject rejects duplicate object keys and any JSON spelling
// that is not the compact, key-sorted representation emitted by the native
// nlohmann::json authority. json.Number preserves the authority's number spelling.
func decodeCanonicalJSONObject(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	value, err := decodeUniqueJSONValue(decoder)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("descriptor contains more than one JSON value")
		}
		return nil, err
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("descriptor is not an object")
	}
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	canonicalBytes := bytes.TrimSuffix(canonical.Bytes(), []byte("\n"))
	if !bytes.Equal(raw, canonicalBytes) {
		return nil, fmt.Errorf("descriptor JSON is not canonical")
	}
	return object, nil
}

func decodeUniqueJSONValue(decoder *json.Decoder) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delimiter, compound := token.(json.Delim)
	if !compound {
		return token, nil
	}
	switch delimiter {
	case '{':
		object := make(map[string]any)
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return nil, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, fmt.Errorf("object key is not a string")
			}
			if _, duplicate := object[key]; duplicate {
				return nil, fmt.Errorf("duplicate object key %q", key)
			}
			value, err := decodeUniqueJSONValue(decoder)
			if err != nil {
				return nil, err
			}
			object[key] = value
		}
		end, err := decoder.Token()
		if err != nil || end != json.Delim('}') {
			return nil, fmt.Errorf("unterminated object")
		}
		return object, nil
	case '[':
		array := make([]any, 0)
		for decoder.More() {
			value, err := decodeUniqueJSONValue(decoder)
			if err != nil {
				return nil, err
			}
			array = append(array, value)
		}
		end, err := decoder.Token()
		if err != nil || end != json.Delim(']') {
			return nil, fmt.Errorf("unterminated array")
		}
		return array, nil
	default:
		return nil, fmt.Errorf("unexpected JSON delimiter")
	}
}

func exactObjectKeys(object map[string]any, required, optional []string) bool {
	allowed := make(map[string]bool, len(required)+len(optional))
	for _, key := range required {
		allowed[key] = true
		if _, present := object[key]; !present {
			return false
		}
	}
	for _, key := range optional {
		allowed[key] = true
	}
	if len(object) < len(required) || len(object) > len(allowed) {
		return false
	}
	for key := range object {
		if !allowed[key] {
			return false
		}
	}
	return true
}

func boundedIdentity(value any, allowWildcard bool) (string, bool) {
	identity, ok := value.(string)
	if !ok || identity == "" || len(identity) > 192 {
		return "", false
	}
	if allowWildcard && identity == "*" {
		return identity, true
	}
	for index, character := range []byte(identity) {
		letter := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z'
		digit := character >= '0' && character <= '9'
		if (!letter && !digit && !strings.ContainsRune("_-.:", rune(character))) ||
			(index == 0 && !letter && !digit) {
			return "", false
		}
	}
	return identity, true
}

func stringArray(value any, allowEmpty, allowWildcard bool) ([]string, bool) {
	raw, ok := value.([]any)
	if !ok || len(raw) > 256 || (!allowEmpty && len(raw) == 0) {
		return nil, false
	}
	values := make([]string, 0, len(raw))
	for _, item := range raw {
		identity, ok := trainingSymbolicIdentity(item, allowWildcard, false)
		if !ok {
			return nil, false
		}
		values = append(values, identity)
	}
	if !sort.StringsAreSorted(values) {
		return nil, false
	}
	for index := 1; index < len(values); index++ {
		if values[index] == values[index-1] {
			return nil, false
		}
	}
	return values, true
}

func trainingSymbolicIdentity(value any, allowWildcard, allowLeadingDigit bool) (string, bool) {
	identity, ok := value.(string)
	if !ok || identity == "" || len(identity) > 192 || identity == "*" && !allowWildcard {
		return "", false
	}
	if allowWildcard && identity == "*" {
		return identity, true
	}
	for index, character := range []byte(identity) {
		letter := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z'
		digit := character >= '0' && character <= '9'
		if (!letter && !digit && !strings.ContainsRune("_-.:", rune(character))) ||
			(index == 0 && !letter && !(allowLeadingDigit && digit)) {
			return "", false
		}
	}
	return identity, true
}

var trainingComponentCategoryOrder = map[string]int{
	"optimizer": 0, "parameter_router": 1, "learning_rate_schedule": 2,
	"weight_decay_schedule": 3, "activation": 4, "normalization": 5,
	"objective": 6, "precision": 7, "gradient_clipping": 8,
	"gradient_accumulation": 9, "curriculum": 10, "metric_reducer": 11,
}

func validateTrainingComponentDescriptors(items []any) error {
	if len(items) > 2048 {
		return fmt.Errorf("training component descriptor count exceeds its bound")
	}
	previous := ""
	seen := make(map[string]bool, len(items))
	for _, item := range items {
		component, ok := item.(map[string]any)
		if !ok || !exactObjectKeys(component,
			[]string{"backend", "configuration", "implementation", "key", "model_families", "reference_implementation", "required_capabilities", "state", "state_grade"},
			[]string{"step_domain"}) {
			return fmt.Errorf("training component descriptor has the wrong shape")
		}
		key, ok := component["key"].(map[string]any)
		if !ok || !exactObjectKeys(key, []string{"category", "name", "version"}, nil) {
			return fmt.Errorf("training component key has the wrong shape")
		}
		category, categoryOK := key["category"].(string)
		categoryIndex, categoryKnown := trainingComponentCategoryOrder[category]
		name, nameOK := trainingSymbolicIdentity(key["name"], false, false)
		version, versionOK := trainingSymbolicIdentity(key["version"], false, true)
		if !categoryOK || !categoryKnown || !nameOK || !versionOK {
			return fmt.Errorf("training component key has malformed basic fields")
		}
		identity := fmt.Sprintf("%02d\x00%s\x00%s", categoryIndex, name, version)
		if seen[identity] || previous != "" && identity <= previous {
			return fmt.Errorf("training component identities are duplicate or noncanonical")
		}
		seen[identity], previous = true, identity
		if _, ok := trainingSymbolicIdentity(component["implementation"], false, false); !ok {
			return fmt.Errorf("training component implementation is malformed")
		}
		backend, ok := component["backend"].(string)
		if !ok || !oneOf(backend, "python", "native", "cuda_extension", "runtime_builtin") {
			return fmt.Errorf("training component backend has the wrong type or value")
		}
		modelFamilies, ok := stringArray(component["model_families"], false, true)
		if !ok || len(modelFamilies) > 1 && modelFamilies[0] == "*" {
			return fmt.Errorf("training component model families are not canonical")
		}
		if _, ok := stringArray(component["required_capabilities"], true, false); !ok {
			return fmt.Errorf("training component capabilities are not canonical")
		}
		if _, err := validateTrainingFields(component["configuration"], false); err != nil {
			return err
		}
		stateFields, err := validateTrainingFields(component["state"], true)
		if err != nil {
			return err
		}
		grade, gradeOK := component["state_grade"].(string)
		reference, referenceOK := component["reference_implementation"].(bool)
		if !gradeOK || !oneOf(grade, "stateless", "compatible", "exact") || !referenceOK {
			return fmt.Errorf("training component state metadata has the wrong basic type")
		}
		if (grade == "stateless") != (stateFields == 0) {
			return fmt.Errorf("training component state grade disagrees with its state schema")
		}
		if reference && backend == "cuda_extension" {
			return fmt.Errorf("CUDA training components cannot be reference implementations")
		}
		scheduled := category == "learning_rate_schedule" || category == "weight_decay_schedule" ||
			category == "gradient_accumulation" || category == "curriculum"
		_, hasStepDomain := component["step_domain"]
		if scheduled != hasStepDomain {
			return fmt.Errorf("training component step domain presence disagrees with its category")
		}
		if step, present := component["step_domain"]; present {
			value, ok := step.(string)
			if !ok || !oneOf(value, "microbatch", "optimizer_step", "sample", "token", "epoch", "wall_time") {
				return fmt.Errorf("training component step domain is malformed")
			}
		}
	}
	return nil
}

func validateTrainingFields(value any, state bool) (int, error) {
	fields, ok := value.([]any)
	if !ok || len(fields) > 128 {
		return 0, fmt.Errorf("training component fields have the wrong type or exceed their bound")
	}
	previous := ""
	for _, item := range fields {
		field, ok := item.(map[string]any)
		optional := []string{"default", "description", "maximum", "minimum", "unit", "values"}
		if state {
			optional = []string{"description"}
		}
		if !ok || !exactObjectKeys(field, []string{"name", "required", "type"}, optional) {
			return 0, fmt.Errorf("training component field has the wrong shape")
		}
		name, nameOK := trainingSymbolicIdentity(field["name"], false, false)
		fieldType, typeOK := field["type"].(string)
		_, requiredOK := field["required"].(bool)
		if !nameOK || !typeOK || !oneOf(fieldType, "boolean", "integer", "number", "string", "enumeration") || !requiredOK ||
			(previous != "" && name <= previous) {
			return 0, fmt.Errorf("training component field identity, order, or basic type is invalid")
		}
		previous = name
		minimum, hasMinimum := finiteJSONNumber(field["minimum"])
		maximum, hasMaximum := finiteJSONNumber(field["maximum"])
		for _, bound := range []string{"minimum", "maximum"} {
			if raw, present := field[bound]; present {
				if _, ok := finiteJSONNumber(raw); !ok {
					return 0, fmt.Errorf("training component numeric bound has the wrong type or is nonfinite")
				}
			}
		}
		if hasMinimum && hasMaximum && minimum > maximum {
			return 0, fmt.Errorf("training component numeric bounds are inverted")
		}
		if (hasMinimum || hasMaximum) && fieldType != "integer" && fieldType != "number" {
			return 0, fmt.Errorf("nonnumeric training component field declares numeric bounds")
		}
		var enumValues []string
		if raw, present := field["values"]; present {
			values, ok := raw.([]any)
			if !ok || fieldType != "enumeration" || len(values) == 0 || len(values) > 256 {
				return 0, fmt.Errorf("training component enum values have the wrong type or bound")
			}
			enumValues = make([]string, 0, len(values))
			for _, value := range values {
				text, ok := value.(string)
				if !ok || len(text) > 4096 {
					return 0, fmt.Errorf("training component enum value has the wrong type or bound")
				}
				enumValues = append(enumValues, text)
			}
			if !sort.StringsAreSorted(enumValues) {
				return 0, fmt.Errorf("training component enum values are not canonical")
			}
			for index := 1; index < len(enumValues); index++ {
				if enumValues[index] == enumValues[index-1] {
					return 0, fmt.Errorf("training component enum values are not unique")
				}
			}
		} else if fieldType == "enumeration" {
			return 0, fmt.Errorf("training component enum field omits its values")
		}
		for _, text := range []string{"description", "unit"} {
			if raw, present := field[text]; present {
				value, ok := raw.(string)
				if !ok || text == "description" && len(value) > 2048 {
					return 0, fmt.Errorf("training component field text has the wrong type or bound")
				}
				if text == "unit" {
					if _, ok := trainingSymbolicIdentity(value, false, false); !ok {
						return 0, fmt.Errorf("training component field unit is malformed")
					}
				}
			}
		}
		if defaultValue, present := field["default"]; present {
			if !trainingValueHasType(fieldType, defaultValue) {
				return 0, fmt.Errorf("training component field default has the wrong type or bound")
			}
			if numeric, numericValue := finiteJSONNumber(defaultValue); numericValue &&
				(hasMinimum && numeric < minimum || hasMaximum && numeric > maximum) {
				return 0, fmt.Errorf("training component field default violates a numeric bound")
			}
			if fieldType == "enumeration" {
				text := defaultValue.(string)
				index := sort.SearchStrings(enumValues, text)
				if index == len(enumValues) || enumValues[index] != text {
					return 0, fmt.Errorf("training component field default is outside its enum")
				}
			}
		}
	}
	return len(fields), nil
}

func finiteJSONNumber(value any) (float64, bool) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(string(number), 64)
	return parsed, err == nil && !math.IsInf(parsed, 0) && !math.IsNaN(parsed)
}

func trainingValueHasType(fieldType string, value any) bool {
	switch fieldType {
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "integer":
		number, ok := value.(json.Number)
		if !ok || strings.ContainsAny(string(number), ".eE") {
			return false
		}
		if _, err := strconv.ParseInt(string(number), 10, 64); err == nil {
			return true
		}
		_, err := strconv.ParseUint(string(number), 10, 64)
		return err == nil
	case "number":
		_, ok := finiteJSONNumber(value)
		return ok
	case "string", "enumeration":
		text, ok := value.(string)
		return ok && len(text) <= 4096
	default:
		return false
	}
}

var operationRuntimeOrder = map[string]int{
	"builtin": 0, "python_worker": 1, "native_worker": 2, "external_worker": 3,
}

func validateOperationDescriptors(items []any) error {
	if len(items) == 0 || len(items) > 4096 {
		return fmt.Errorf("operation descriptor count exceeds its bound")
	}
	previous := ""
	selectors := make(map[string]bool, len(items))
	identities := make(map[string]bool, len(items))
	for _, item := range items {
		operation, ok := item.(map[string]any)
		if !ok || !exactObjectKeys(operation,
			[]string{"authoring", "code_fingerprint", "effect", "idempotency", "key", "lifecycle", "required_capabilities"},
			[]string{"training_composition"}) {
			return fmt.Errorf("operation descriptor has the wrong shape")
		}
		key, ok := operation["key"].(map[string]any)
		if !ok || !exactObjectKeys(key, []string{"adapter", "contract", "operation", "runtime", "version"}, nil) {
			return fmt.Errorf("operation descriptor key has the wrong shape")
		}
		adapter, adapterOK := boundedIdentity(key["adapter"], false)
		version, versionOK := boundedIdentity(key["version"], false)
		name, nameOK := boundedIdentity(key["operation"], false)
		contract, contractOK := boundedIdentity(key["contract"], false)
		runtime, runtimeOK := key["runtime"].(string)
		runtimeIndex, runtimeKnown := operationRuntimeOrder[runtime]
		if !adapterOK || !versionOK || !nameOK || !contractOK || !runtimeOK || !runtimeKnown {
			return fmt.Errorf("operation descriptor identity has malformed basic fields")
		}
		selector := fmt.Sprintf("%s\x00%s\x00%02d\x00%s", adapter, version, runtimeIndex, name)
		identity := selector + "\x00" + contract
		if selectors[selector] || identities[identity] || previous != "" && identity <= previous {
			return fmt.Errorf("operation descriptor identities are duplicate or noncanonical")
		}
		selectors[selector], identities[identity], previous = true, true, identity
		effect, effectOK := operation["effect"].(string)
		idempotency, idempotencyOK := operation["idempotency"].(string)
		fingerprint, fingerprintOK := operation["code_fingerprint"].(string)
		if !effectOK || !oneOf(effect, "read_only", "workspace_write", "process", "resource", "external") ||
			!idempotencyOK || !oneOf(idempotency, "replay_safe", "receipt_required", "at_most_once") ||
			!fingerprintOK || len(fingerprint) > 256 {
			return fmt.Errorf("operation descriptor authority fields have the wrong basic type")
		}
		if _, ok := canonicalBoundedStrings(operation["required_capabilities"], true); !ok {
			return fmt.Errorf("operation capabilities are not canonical")
		}
		if err := validateOperationLifecycle(operation["lifecycle"]); err != nil {
			return err
		}
		if contract, present := operation["training_composition"]; present {
			if err := validateOperationTrainingContract(contract); err != nil {
				return err
			}
		}
		if err := validateOperationAuthoring(operation["authoring"]); err != nil {
			return err
		}
	}
	return nil
}

func validateOperationLifecycle(value any) error {
	lifecycle, ok := value.(map[string]any)
	required := []string{"checkpoint_now", "compile", "graceful_stop", "pause_keep_resources", "pause_release_resources", "profile", "qualify", "resume_grade", "stateful", "warmup"}
	if !ok || !exactObjectKeys(lifecycle, required, nil) {
		return fmt.Errorf("operation lifecycle has the wrong shape")
	}
	for _, field := range required {
		if field == "resume_grade" {
			grade, ok := lifecycle[field].(string)
			if !ok || !oneOf(grade, "none", "restart_only", "terminal_checkpoint", "compatible", "exact") {
				return fmt.Errorf("operation resume grade is malformed")
			}
		} else if _, ok := lifecycle[field].(bool); !ok {
			return fmt.Errorf("operation lifecycle flag has the wrong type")
		}
	}
	return nil
}

func validateOperationTrainingContract(value any) error {
	contract, ok := value.(map[string]any)
	if !ok || !exactObjectKeys(contract, []string{"model_family", "slots"}, nil) {
		return fmt.Errorf("operation training contract has the wrong shape")
	}
	if _, ok := boundedIdentity(contract["model_family"], false); !ok {
		return fmt.Errorf("operation model family is malformed")
	}
	slots, ok := contract["slots"].(map[string]any)
	if !ok || len(slots) == 0 || len(slots) > 64 {
		return fmt.Errorf("operation training slots have the wrong type or bound")
	}
	for name, value := range slots {
		if _, ok := boundedIdentity(name, false); !ok {
			return fmt.Errorf("operation training slot name is malformed")
		}
		category, ok := value.(string)
		if !ok {
			return fmt.Errorf("operation training slot category has the wrong type")
		}
		if _, known := trainingComponentCategoryOrder[category]; !known {
			return fmt.Errorf("operation training slot category is unknown")
		}
	}
	return nil
}

func validateOperationAuthoring(value any) error {
	authoring, ok := value.(map[string]any)
	if !ok || !exactObjectKeys(authoring, []string{"inputs", "outputs"}, nil) {
		return fmt.Errorf("operation authoring declaration has the wrong shape")
	}
	for _, direction := range []string{"inputs", "outputs"} {
		ports, ok := authoring[direction].(map[string]any)
		if !ok || len(ports) > 64 {
			return fmt.Errorf("operation port declaration has the wrong type or bound")
		}
		for name, raw := range ports {
			if _, ok := boundedIdentity(name, false); !ok {
				return fmt.Errorf("operation port name is malformed")
			}
			port, ok := raw.(map[string]any)
			if !ok || !exactObjectKeys(port, []string{"required", "type"}, []string{"artifact_schema", "artifact_type", "description"}) {
				return fmt.Errorf("operation port descriptor has the wrong shape")
			}
			portType, typeOK := port["type"].(string)
			_, requiredOK := port["required"].(bool)
			if !typeOK || !oneOf(portType, "string", "integer", "number", "boolean", "object", "artifact") || !requiredOK {
				return fmt.Errorf("operation port has the wrong basic type")
			}
			if direction == "outputs" && portType != "artifact" {
				return fmt.Errorf("operation outputs must publish artifacts")
			}
			for _, field := range []string{"artifact_schema", "artifact_type", "description"} {
				if raw, present := port[field]; present {
					text, ok := raw.(string)
					if !ok {
						return fmt.Errorf("operation port metadata has the wrong type")
					}
					if field == "artifact_type" && !oneOf(text, "path", "checkpoint", "dataset", "image_gallery", "metrics", "report", "opaque") {
						return fmt.Errorf("operation artifact port type is unknown")
					}
					if field == "artifact_schema" && (text == "" || len(text) > 512) ||
						field == "description" && len(text) > 4<<10 {
						return fmt.Errorf("operation port metadata exceeds its bound")
					}
				}
			}
			if portType != "artifact" && (port["artifact_type"] != nil || port["artifact_schema"] != nil) {
				if _, present := port["artifact_type"]; present {
					return fmt.Errorf("non-artifact operation port narrows an artifact type")
				}
				if _, present := port["artifact_schema"]; present {
					return fmt.Errorf("non-artifact operation port narrows an artifact schema")
				}
			}
		}
	}
	return nil
}

func canonicalBoundedStrings(value any, allowEmpty bool) ([]string, bool) {
	raw, ok := value.([]any)
	if !ok || len(raw) > 256 || (!allowEmpty && len(raw) == 0) {
		return nil, false
	}
	values := make([]string, 0, len(raw))
	for _, item := range raw {
		text, ok := item.(string)
		if !ok || text == "" || len(text) > 256 {
			return nil, false
		}
		values = append(values, text)
	}
	if !sort.StringsAreSorted(values) {
		return nil, false
	}
	for index := 1; index < len(values); index++ {
		if values[index] == values[index-1] {
			return nil, false
		}
	}
	return values, true
}

func oneOf(value string, choices ...string) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}
