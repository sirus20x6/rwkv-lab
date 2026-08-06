package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	trainvmstore "trainboard/internal/trainvm"
)

const trainVMDescriptorLimit = 1 << 20

type trainVMDescriptorSpec struct {
	provider               string
	version                string
	apiVersion             string
	listField              string
	requiredEnvelopeFields []string
	optionalEnvelopeFields []string
	maximumBytes           int
	validate               func([]any) error
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
	trainVMRecipeProfilesDescriptor = trainVMDescriptorSpec{
		provider: "trainvm.recipe-profiles", version: "1.0.0",
		apiVersion: "trainvm.recipe-profiles/v1", listField: "recipes",
		requiredEnvelopeFields: []string{"default_registry_path", "registry_digest", "registry_path"},
		maximumBytes:           16 << 20,
		validate:               validateRecipeProfileDescriptors,
	}
)

func (s *Server) handleTrainVMTrainingComponents(w http.ResponseWriter, r *http.Request) {
	s.handleTrainVMDescriptor(w, r, trainVMTrainingComponentsDescriptor)
}

func (s *Server) handleTrainVMOperations(w http.ResponseWriter, r *http.Request) {
	s.handleTrainVMDescriptor(w, r, trainVMOperationsDescriptor)
}

func (s *Server) handleTrainVMRecipeProfiles(w http.ResponseWriter, r *http.Request) {
	s.handleTrainVMDescriptor(w, r, trainVMRecipeProfilesDescriptor)
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
	maximum := spec.maximumBytes
	if maximum == 0 {
		maximum = trainVMDescriptorLimit
	}
	if len(raw) == 0 || len(raw) > maximum {
		return nil, fmt.Errorf("descriptor size is outside the authority bound")
	}
	decoded, err := decodeCanonicalJSONObject(raw)
	if err != nil {
		return nil, err
	}
	required := append([]string{"api_version", spec.listField}, spec.requiredEnvelopeFields...)
	if !exactObjectKeys(decoded, required, spec.optionalEnvelopeFields) {
		return nil, fmt.Errorf("descriptor top-level envelope is not exact")
	}
	apiVersion, ok := decoded["api_version"].(string)
	if !ok || apiVersion != spec.apiVersion {
		return nil, fmt.Errorf("descriptor api_version is not the requested contract")
	}
	if registryPath, present := decoded["registry_path"]; present {
		path, ok := canonicalDescriptorPath(registryPath)
		defaultPath, defaultOK := canonicalDescriptorPath(decoded["default_registry_path"])
		registryDigest, digestOK := decoded["registry_digest"].(string)
		if !ok || !defaultOK || defaultPath != path || !digestOK || !canonicalSHA256(registryDigest) {
			return nil, fmt.Errorf("descriptor recipe registry authority metadata is invalid")
		}
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
	if registryDigest, present := decoded["registry_digest"].(string); present {
		canonicalRegistry, err := canonicalJSONBytes(map[string]any{
			"api_version":  decoded["api_version"],
			spec.listField: items,
		})
		if err != nil || registryDigest != fmt.Sprintf("sha256:%x", sha256.Sum256(canonicalRegistry)) {
			return nil, fmt.Errorf("descriptor recipe registry digest does not match its canonical profiles")
		}
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(raw))
	if result.SchemaHash != digest {
		return nil, fmt.Errorf("descriptor identity does not match its canonical bytes")
	}
	return decoded, nil
}

func canonicalDescriptorPath(value any) (string, bool) {
	path, ok := value.(string)
	return path, ok && path != "" && path != string(filepath.Separator) && len(path) <= 4096 &&
		filepath.IsAbs(path) && filepath.Clean(path) == path && !strings.Contains(path, "\x00")
}

func canonicalSHA256(value string) bool {
	if len(value) != 71 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, character := range value[7:] {
		if !(character >= '0' && character <= '9' || character >= 'a' && character <= 'f') {
			return false
		}
	}
	return true
}

func validateRecipeProfileDescriptors(items []any) error {
	if len(items) > 256 {
		return fmt.Errorf("recipe profile count exceeds its bound")
	}
	previous := ""
	for _, item := range items {
		profile, ok := item.(map[string]any)
		if !ok || !exactObjectKeys(profile,
			[]string{"key", "overrides", "template_document"},
			[]string{"compatibility", "description"}) {
			return fmt.Errorf("recipe profile descriptor has the wrong shape")
		}
		key, ok := profile["key"].(map[string]any)
		if !ok || !exactObjectKeys(key, []string{"name", "version"}, nil) {
			return fmt.Errorf("recipe profile key has the wrong shape")
		}
		name, nameOK := trainingSymbolicIdentity(key["name"], false, false)
		version, versionOK := trainingSymbolicIdentity(key["version"], false, true)
		identity := name + "\x00" + version
		if !nameOK || !versionOK || previous != "" && identity <= previous {
			return fmt.Errorf("recipe profile identities are malformed or noncanonical")
		}
		previous = identity
		if description, present := profile["description"]; present {
			text, ok := description.(string)
			if !ok || text == "" || len(text) > 4096 {
				return fmt.Errorf("recipe profile description is malformed")
			}
		}
		if _, ok := profile["template_document"].(map[string]any); !ok {
			return fmt.Errorf("recipe template document is not an object")
		}
		rawOverrides, ok := profile["overrides"].([]any)
		if !ok || len(rawOverrides) > 256 {
			return fmt.Errorf("recipe override fields have the wrong type or exceed their bound")
		}
		fieldNames := make(map[string]bool, len(rawOverrides))
		fields := make(map[string]map[string]any, len(rawOverrides))
		previousField := ""
		for _, raw := range rawOverrides {
			field, ok := raw.(map[string]any)
			if !ok || !exactObjectKeys(field,
				[]string{"domain", "name", "required", "target", "type"},
				[]string{"description", "maximum", "minimum", "values"}) {
				return fmt.Errorf("recipe override field has the wrong shape")
			}
			fieldName, nameOK := recipeFieldName(field["name"])
			_, domainOK := trainingSymbolicIdentity(field["domain"], false, false)
			fieldType, typeOK := field["type"].(string)
			target, targetOK := field["target"].(string)
			_, requiredOK := field["required"].(bool)
			if !nameOK || !domainOK || !typeOK ||
				!oneOf(fieldType, "boolean", "integer", "number", "string", "path", "enumeration") ||
				!targetOK || target == "" || len(target) > 4096 || target[0] != '/' || !requiredOK ||
				previousField != "" && fieldName <= previousField {
				return fmt.Errorf("recipe override identity, target, or type is invalid")
			}
			previousField, fieldNames[fieldName] = fieldName, true
			fields[fieldName] = field
			minimum, hasMinimum := finiteJSONNumber(field["minimum"])
			maximum, hasMaximum := finiteJSONNumber(field["maximum"])
			if (hasMinimum || hasMaximum) && fieldType != "integer" && fieldType != "number" ||
				hasMinimum && hasMaximum && minimum > maximum {
				return fmt.Errorf("recipe override bounds are invalid")
			}
			values, hasValues := field["values"]
			if fieldType == "enumeration" {
				rawValues, ok := values.([]any)
				if !hasValues || !ok || len(rawValues) == 0 || len(rawValues) > 256 {
					return fmt.Errorf("recipe enumeration values are invalid")
				}
				seenValues := make(map[string]bool, len(rawValues))
				for _, value := range rawValues {
					text, stringOK := value.(string)
					encoded, err := json.Marshal(value)
					if !stringOK || text == "" || len(text) > 192 || err != nil || seenValues[string(encoded)] {
						return fmt.Errorf("recipe enumeration values are duplicate or unencodable")
					}
					seenValues[string(encoded)] = true
				}
				for index := 1; index < len(rawValues); index++ {
					if rawValues[index-1].(string) >= rawValues[index].(string) {
						return fmt.Errorf("recipe enumeration values are not canonical")
					}
				}
			} else if hasValues {
				return fmt.Errorf("non-enumeration recipe override carries values")
			}
		}
		if compatibility, present := profile["compatibility"]; present {
			rules, ok := compatibility.([]any)
			if !ok || len(rules) > 128 {
				return fmt.Errorf("recipe compatibility rules are invalid")
			}
			for _, raw := range rules {
				rule, ok := raw.(map[string]any)
				if !ok || !exactObjectKeys(rule, []string{"allowed", "fields"}, []string{"description"}) {
					return fmt.Errorf("recipe compatibility rule has the wrong shape")
				}
				rawFields, fieldsOK := rule["fields"].([]any)
				allowed, allowedOK := rule["allowed"].([]any)
				if !fieldsOK || !allowedOK || len(rawFields) == 0 || len(rawFields) > 32 ||
					len(allowed) == 0 || len(allowed) > 1024 {
					return fmt.Errorf("recipe compatibility rule exceeds its bounds")
				}
				seenFields := make(map[string]bool, len(rawFields))
				fieldOrder := make([]string, 0, len(rawFields))
				for _, value := range rawFields {
					name, ok := recipeFieldName(value)
					if !ok || !fieldNames[name] || seenFields[name] {
						return fmt.Errorf("recipe compatibility rule names an invalid field")
					}
					seenFields[name] = true
					fieldOrder = append(fieldOrder, name)
				}
				for _, value := range allowed {
					tuple, ok := value.([]any)
					if !ok || len(tuple) != len(rawFields) {
						return fmt.Errorf("recipe compatibility tuple has the wrong arity")
					}
					for index, tupleValue := range tuple {
						if !recipeValueValid(fields[fieldOrder[index]], tupleValue) {
							return fmt.Errorf("recipe compatibility tuple value has the wrong type or bound")
						}
					}
				}
			}
		}
	}
	return nil
}

func recipeValueValid(field map[string]any, value any) bool {
	fieldType, _ := field["type"].(string)
	switch fieldType {
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "integer":
		number, ok := finiteJSONNumber(value)
		if !ok || math.Trunc(number) != number {
			return false
		}
		minimum, hasMinimum := finiteJSONNumber(field["minimum"])
		maximum, hasMaximum := finiteJSONNumber(field["maximum"])
		return (!hasMinimum || number >= minimum) && (!hasMaximum || number <= maximum)
	case "number":
		number, ok := finiteJSONNumber(value)
		if !ok {
			return false
		}
		minimum, hasMinimum := finiteJSONNumber(field["minimum"])
		maximum, hasMaximum := finiteJSONNumber(field["maximum"])
		return (!hasMinimum || number >= minimum) && (!hasMaximum || number <= maximum)
	case "string", "path":
		text, ok := value.(string)
		return ok && text != "" && len(text) <= 4096
	case "enumeration":
		text, ok := value.(string)
		if !ok {
			return false
		}
		for _, candidate := range field["values"].([]any) {
			if candidate == text {
				return true
			}
		}
	}
	return false
}

func recipeFieldName(value any) (string, bool) {
	name, ok := value.(string)
	if !ok || name == "" || len(name) > 192 || strings.HasPrefix(name, ".") || strings.HasSuffix(name, ".") {
		return "", false
	}
	for _, part := range strings.Split(name, ".") {
		if _, ok := trainingSymbolicIdentity(part, false, false); !ok {
			return "", false
		}
	}
	return name, true
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
	canonicalBytes, err := canonicalJSONBytes(value)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(raw, canonicalBytes) {
		return nil, fmt.Errorf("descriptor JSON is not canonical")
	}
	return object, nil
}

func canonicalJSONBytes(value any) ([]byte, error) {
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(canonical.Bytes(), []byte("\n")), nil
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
	"model_loader": 0, "trainability": 1, "data_source": 2,
	"sample_processor": 3, "sample_mapper": 4, "collator": 5,
	"sampler": 6, "batching": 7, "split_selector": 8,
	"optimizer": 9, "parameter_router": 10, "learning_rate_schedule": 11,
	"weight_decay_schedule": 12, "activation": 13, "normalization": 14,
	"objective": 15, "precision": 16, "gradient_clipping": 17,
	"gradient_accumulation": 18, "curriculum": 19, "metric_reducer": 20,
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
