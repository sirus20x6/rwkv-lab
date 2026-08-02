package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
)

const trainVMTrainingLockLimit = 1 << 20

func validateTrainingComponentPreview(canonicalPlan any, digest, lock string) error {
	digest = strings.TrimSpace(digest)
	if digest == "" && lock == "" {
		if planUsesTrainingComposition(canonicalPlan) {
			return fmt.Errorf("authority preview omitted the required training-component lock")
		}
		return nil
	}
	if digest == "" || lock == "" || len(lock) > trainVMTrainingLockLimit {
		return fmt.Errorf("authority preview returned an incomplete training-component lock")
	}
	document, err := decodeCanonicalJSONObject([]byte(lock))
	if err != nil {
		return fmt.Errorf("authority preview returned a malformed training-component lock: %w", err)
	}
	want := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(lock)))
	if digest != want {
		return fmt.Errorf("authority preview training-component lock identity does not match its canonical bytes")
	}
	return validateTrainingComponentLockDocument(canonicalPlan, document)
}

func planUsesTrainingComposition(canonicalPlan any) bool {
	compositions, err := planTrainingCompositions(canonicalPlan)
	return err == nil && len(compositions) != 0
}

func planTrainingCompositions(canonicalPlan any) (map[string]map[string]any, error) {
	compositions := make(map[string]map[string]any)
	plan, ok := canonicalPlan.(map[string]any)
	if !ok {
		return compositions, nil
	}
	spec, ok := plan["spec"].(map[string]any)
	if !ok {
		return compositions, nil
	}
	workflow, ok := spec["workflow"].(map[string]any)
	if !ok {
		return compositions, nil
	}
	nodes, ok := workflow["nodes"].(map[string]any)
	if !ok {
		return compositions, nil
	}
	for name, rawNode := range nodes {
		node, ok := rawNode.(map[string]any)
		if !ok {
			continue
		}
		invocation, ok := node["invoke"].(map[string]any)
		if !ok {
			continue
		}
		if training, present := invocation["training"]; present && training != nil {
			composition, ok := training.(map[string]any)
			if !ok || !exactObjectKeys(composition, []string{"components", "model_family"}, nil) {
				return nil, fmt.Errorf("canonical plan training composition is malformed")
			}
			if _, ok := composition["model_family"].(string); !ok {
				return nil, fmt.Errorf("canonical plan training model family is malformed")
			}
			if _, ok := composition["components"].(map[string]any); !ok {
				return nil, fmt.Errorf("canonical plan training components are malformed")
			}
			compositions[name] = composition
		}
	}
	return compositions, nil
}

func validateTrainingComponentLockDocument(canonicalPlan any, document map[string]any) error {
	if !exactObjectKeys(document, []string{"api_version", "nodes", "registry_digest"}, nil) {
		return fmt.Errorf("authority preview returned a non-exact training-component lock envelope")
	}
	apiVersion, ok := document["api_version"].(string)
	if !ok || apiVersion != "trainvm.training-component-lock/v1" {
		return fmt.Errorf("authority preview returned the wrong training-component lock version")
	}
	registryDigest, ok := document["registry_digest"].(string)
	if !ok || !validCanonicalSHA256(registryDigest) {
		return fmt.Errorf("authority preview returned a malformed training-component registry identity")
	}
	lockedNodes, ok := document["nodes"].(map[string]any)
	if !ok {
		return fmt.Errorf("authority preview training-component lock nodes have the wrong type")
	}
	plannedNodes, err := planTrainingCompositions(canonicalPlan)
	if err != nil {
		return err
	}
	if len(lockedNodes) != len(plannedNodes) {
		return fmt.Errorf("authority preview training-component lock node membership disagrees with the canonical plan")
	}
	for nodeName, planned := range plannedNodes {
		rawLocked, present := lockedNodes[nodeName]
		locked, ok := rawLocked.(map[string]any)
		if !present || !ok {
			return fmt.Errorf("authority preview training-component lock omits canonical plan node %q", nodeName)
		}
		if err := validateResolvedTrainingComposition(planned, locked, registryDigest); err != nil {
			return fmt.Errorf("authority preview training-component lock node %q: %w", nodeName, err)
		}
	}
	return nil
}

func validateResolvedTrainingComposition(planned, locked map[string]any, registryDigest string) error {
	if !exactObjectKeys(locked,
		[]string{"api_version", "components", "composition_digest", "model_family", "registry_digest"}, nil) {
		return fmt.Errorf("resolved composition envelope is not exact")
	}
	apiVersion, apiOK := locked["api_version"].(string)
	modelFamily, familyOK := locked["model_family"].(string)
	plannedFamily, plannedFamilyOK := planned["model_family"].(string)
	nodeRegistry, registryOK := locked["registry_digest"].(string)
	compositionDigest, digestOK := locked["composition_digest"].(string)
	if !apiOK || apiVersion != "trainvm.resolved-training-composition/v1" ||
		!familyOK || !plannedFamilyOK || modelFamily != plannedFamily ||
		!registryOK || nodeRegistry != registryDigest ||
		!digestOK || !validCanonicalSHA256(compositionDigest) {
		return fmt.Errorf("resolved composition identities disagree with the canonical plan or registry")
	}
	plannedComponents, plannedOK := planned["components"].(map[string]any)
	lockedComponents, lockedOK := locked["components"].(map[string]any)
	if !plannedOK || !lockedOK || len(plannedComponents) != len(lockedComponents) {
		return fmt.Errorf("resolved component membership disagrees with the canonical plan")
	}
	for slot, rawSelection := range plannedComponents {
		selection, selectionOK := rawSelection.(map[string]any)
		rawResolved, present := lockedComponents[slot]
		resolved, resolvedOK := rawResolved.(map[string]any)
		if !selectionOK || !present || !resolvedOK ||
			!exactObjectKeys(selection, []string{"configuration", "key"}, nil) ||
			!exactObjectKeys(resolved, []string{"configuration", "descriptor", "descriptor_digest"}, nil) {
			return fmt.Errorf("resolved component slot %q has the wrong shape", slot)
		}
		descriptor, descriptorOK := resolved["descriptor"].(map[string]any)
		if !descriptorOK || validateTrainingComponentDescriptors([]any{descriptor}) != nil ||
			!reflect.DeepEqual(selection["key"], descriptor["key"]) {
			return fmt.Errorf("resolved component slot %q selects a different descriptor", slot)
		}
		families, familiesOK := descriptor["model_families"].([]any)
		compatible := false
		for _, rawFamily := range families {
			family, ok := rawFamily.(string)
			compatible = compatible || ok && (family == "*" || family == modelFamily)
		}
		if !familiesOK || !compatible {
			return fmt.Errorf("resolved component slot %q is incompatible with the plan model family", slot)
		}
		descriptorDigest, digestOK := resolved["descriptor_digest"].(string)
		descriptorBytes, marshalErr := marshalCanonicalJSON(descriptor)
		if !digestOK || marshalErr != nil || descriptorDigest !=
			fmt.Sprintf("sha256:%x", sha256.Sum256(descriptorBytes)) {
			return fmt.Errorf("resolved component slot %q descriptor digest is incoherent", slot)
		}
		requestedConfiguration, requestedOK := selection["configuration"].(map[string]any)
		resolvedConfiguration, configurationOK := resolved["configuration"].(map[string]any)
		if !requestedOK || !configurationOK {
			return fmt.Errorf("resolved component slot %q configuration has the wrong type", slot)
		}
		expectedConfiguration, err := resolveExpectedTrainingConfiguration(
			descriptor, requestedConfiguration)
		if err != nil || !jsonEquivalent(expectedConfiguration, resolvedConfiguration) {
			return fmt.Errorf("resolved component slot %q configuration is not the deterministic descriptor resolution", slot)
		}
	}
	body := map[string]any{
		"api_version": locked["api_version"], "components": locked["components"],
		"model_family": locked["model_family"], "registry_digest": locked["registry_digest"],
	}
	bodyBytes, err := marshalCanonicalJSON(body)
	if err != nil || compositionDigest != fmt.Sprintf("sha256:%x", sha256.Sum256(bodyBytes)) {
		return fmt.Errorf("resolved composition digest is incoherent")
	}
	return nil
}

func resolveExpectedTrainingConfiguration(descriptor,
	requested map[string]any) (map[string]any, error) {
	rawFields, ok := descriptor["configuration"].([]any)
	if !ok {
		return nil, fmt.Errorf("descriptor configuration schema has the wrong type")
	}
	fields := make(map[string]map[string]any, len(rawFields))
	resolved := make(map[string]any)
	for _, raw := range rawFields {
		field, ok := raw.(map[string]any)
		name, nameOK := field["name"].(string)
		if !ok || !nameOK {
			return nil, fmt.Errorf("descriptor configuration field is malformed")
		}
		fields[name] = field
		if value, present := field["default"]; present {
			resolved[name] = value
		}
	}
	for name, value := range requested {
		field, present := fields[name]
		if !present || !trainingConfigurationValueValid(field, value) {
			return nil, fmt.Errorf("requested configuration violates its descriptor")
		}
		resolved[name] = value
	}
	for name, field := range fields {
		required, ok := field["required"].(bool)
		if !ok {
			return nil, fmt.Errorf("descriptor configuration required flag is malformed")
		}
		if _, present := resolved[name]; required && !present {
			return nil, fmt.Errorf("requested configuration omits required field %q", name)
		}
	}
	return resolved, nil
}

func trainingConfigurationValueValid(field map[string]any, value any) bool {
	fieldType, ok := field["type"].(string)
	if !ok || !previewTrainingValueHasType(fieldType, value) {
		return false
	}
	if numeric, numericValue := previewNumericValue(value); numericValue {
		if minimum, present := finiteJSONNumber(field["minimum"]); present && numeric < minimum {
			return false
		}
		if maximum, present := finiteJSONNumber(field["maximum"]); present && numeric > maximum {
			return false
		}
	}
	if rawValues, present := field["values"]; present {
		values, ok := rawValues.([]any)
		if !ok {
			return false
		}
		for _, allowed := range values {
			if jsonEquivalent(allowed, value) {
				return true
			}
		}
		return false
	}
	return true
}

func previewTrainingValueHasType(fieldType string, value any) bool {
	switch fieldType {
	case "integer":
		number, ok := previewNumericValue(value)
		return ok && math.Trunc(number) == number
	case "number":
		_, ok := previewNumericValue(value)
		return ok
	default:
		return trainingValueHasType(fieldType, value)
	}
}

func previewNumericValue(value any) (float64, bool) {
	if number, ok := finiteJSONNumber(value); ok {
		return number, true
	}
	number, ok := value.(float64)
	return number, ok && !math.IsInf(number, 0) && !math.IsNaN(number)
}

func jsonEquivalent(left, right any) bool {
	if leftNumber, leftOK := previewNumericValue(left); leftOK {
		rightNumber, rightOK := previewNumericValue(right)
		return rightOK && leftNumber == rightNumber
	}
	leftObject, leftOK := left.(map[string]any)
	rightObject, rightOK := right.(map[string]any)
	if leftOK || rightOK {
		if !leftOK || !rightOK || len(leftObject) != len(rightObject) {
			return false
		}
		for key, leftValue := range leftObject {
			rightValue, present := rightObject[key]
			if !present || !jsonEquivalent(leftValue, rightValue) {
				return false
			}
		}
		return true
	}
	leftArray, leftOK := left.([]any)
	rightArray, rightOK := right.([]any)
	if leftOK || rightOK {
		if !leftOK || !rightOK || len(leftArray) != len(rightArray) {
			return false
		}
		for index := range leftArray {
			if !jsonEquivalent(leftArray[index], rightArray[index]) {
				return false
			}
		}
		return true
	}
	return reflect.DeepEqual(left, right)
}

func validCanonicalSHA256(value string) bool {
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, character := range value[len("sha256:"):] {
		if character < '0' || character > '9' && character < 'a' || character > 'f' {
			return false
		}
	}
	return true
}

func marshalCanonicalJSON(value any) ([]byte, error) {
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(encoded.Bytes(), []byte("\n")), nil
}
