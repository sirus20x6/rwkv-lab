#include "trainvm/training_component_registry.hpp"

#include "trainvm/post_training_authority.hpp"
#include "trainvm/rwkv_scratch_profiles.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "trainvm/document.hpp"
#include "trainvm/authority_document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaximumRegistryBytes = 1U << 20U;
constexpr std::size_t kMaximumComponents = 2048U;
constexpr std::size_t kMaximumFields = 128U;
constexpr std::size_t kMaximumValues = 256U;
constexpr std::size_t kMaximumIdentityBytes = 192U;
constexpr std::size_t kMaximumConfigurationBytes = 64U << 10U;
constexpr std::size_t kMaximumCompositionBytes = 512U << 10U;
constexpr std::size_t kMaximumStringValueBytes = 4096U;
constexpr std::size_t kMaximumStringListItems = 256U;

[[noreturn]] void reject(std::string message) {
  throw TrainingComponentResolutionError(std::move(message));
}

bool symbolic_identity(std::string_view value, bool allow_wildcard = false,
                       bool allow_leading_digit = false) {
  if (value.empty() || value.size() > kMaximumIdentityBytes ||
      (!allow_wildcard && value == "*"))
    return false;
  if (allow_wildcard && value == "*") return true;
  if (!((value.front() >= 'a' && value.front() <= 'z') ||
        (value.front() >= 'A' && value.front() <= 'Z') ||
        (allow_leading_digit && value.front() >= '0' &&
         value.front() <= '9')))
    return false;
  return std::ranges::all_of(value, [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.' || character == ':';
  });
}

bool value_has_type(TrainingValueType type, const Json& value) {
  switch (type) {
    case TrainingValueType::boolean:
      return value.is_boolean();
    case TrainingValueType::integer:
      return value.is_number_integer();
    case TrainingValueType::number:
      return value.is_number() &&
             (!value.is_number_float() ||
              std::isfinite(value.get<double>()));
    case TrainingValueType::string:
    case TrainingValueType::path:
    case TrainingValueType::enumeration:
      return value.is_string();
    case TrainingValueType::string_list:
      return value.is_array() &&
             value.size() <= kMaximumStringListItems &&
             std::ranges::all_of(value, [](const Json& item) {
               return item.is_string() &&
                      item.get_ref<const std::string&>().size() <=
                          kMaximumStringValueBytes;
             });
  }
  return false;
}

bool bounded_scalar(const Json& value) {
  if (value.is_string())
    return value.get_ref<const std::string&>().size() <=
           kMaximumStringValueBytes;
  if (value.is_array())
    return value.size() <= kMaximumStringListItems &&
           std::ranges::all_of(value, [](const Json& item) {
             return item.is_string() &&
                    item.get_ref<const std::string&>().size() <=
                        kMaximumStringValueBytes;
           });
  return true;
}

bool sha256_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool parameter_selector(std::string_view value) {
  if (value.empty() || value.size() > 512U || value.front() == '!' ||
      value.find('\0') != std::string_view::npos ||
      value.find('\n') != std::string_view::npos ||
      value.find('\r') != std::string_view::npos)
    return false;
  // fnmatch character classes must be balanced. Other characters are valid
  // parameter-name bytes and are interpreted literally by the worker.
  bool open = false;
  for (const char character : value) {
    if (character == '[') {
      if (open) return false;
      open = true;
    } else if (character == ']') {
      if (!open) return false;
      open = false;
    }
  }
  return !open;
}

bool formatted_string(TrainingStringFormat format, const Json& value) {
  const auto valid = [format](std::string_view item) {
    switch (format) {
      case TrainingStringFormat::parameter_selector:
        return parameter_selector(item);
      case TrainingStringFormat::sha256_digest:
        return sha256_digest(item);
    }
    return false;
  };
  if (value.is_string()) return valid(value.get_ref<const std::string&>());
  return value.is_array() &&
         std::ranges::all_of(value, [&](const Json& item) {
           return item.is_string() &&
                  valid(item.get_ref<const std::string&>());
         });
}

void canonicalize_collection(const TrainingComponentField& field,
                             Json& value) {
  if (field.type != TrainingValueType::string_list || !value.is_array())
    return;
  if (field.collection_semantics == TrainingCollectionSemantics::set) {
    std::vector<std::string> items = value.get<std::vector<std::string>>();
    std::ranges::sort(items);
    if (std::ranges::adjacent_find(items) != items.end())
      reject("training component string-set field contains a duplicate");
    value = std::move(items);
  }
}

bool scheduled(TrainingComponentCategory category) {
  return category == TrainingComponentCategory::learning_rate_schedule ||
         category == TrainingComponentCategory::weight_decay_schedule ||
         category == TrainingComponentCategory::gradient_accumulation ||
         category == TrainingComponentCategory::curriculum;
}

std::vector<std::string> canonical_strings(
    std::vector<std::string> values, std::string_view field,
    bool allow_wildcard = false) {
  if (values.empty() || values.size() > kMaximumValues ||
      std::ranges::any_of(values, [&](const std::string& value) {
        return !symbolic_identity(value, allow_wildcard);
      }))
    reject("training component " + std::string(field) +
           " must be a bounded list of symbolic identities");
  std::ranges::sort(values);
  if (std::ranges::adjacent_find(values) != values.end())
    reject("training component " + std::string(field) +
           " must be unique");
  return values;
}

void validate_field(TrainingComponentField& field, bool state_field) {
  if (!symbolic_identity(field.name) ||
      (field.unit && !symbolic_identity(*field.unit)) ||
      (field.description && field.description->size() > 2048U))
    reject("training component field identity is malformed");
  if (field.minimum && (!std::isfinite(*field.minimum) ||
                        (field.maximum && *field.minimum > *field.maximum)))
    reject("training component field numeric bounds are invalid");
  if (field.maximum && !std::isfinite(*field.maximum))
    reject("training component field numeric bounds are invalid");
  if ((field.minimum || field.maximum) &&
      field.type != TrainingValueType::integer &&
      field.type != TrainingValueType::number)
    reject("only numeric training component fields may have bounds");
  if (field.values) {
    if (field.type != TrainingValueType::enumeration ||
        field.values->empty() || field.values->size() > kMaximumValues ||
        std::ranges::any_of(*field.values, [&](const Json& value) {
          return !value_has_type(field.type, value) ||
                 !bounded_scalar(value);
        }))
      reject("training component enum field values are invalid");
    std::ranges::sort(*field.values);
    if (std::ranges::adjacent_find(*field.values) != field.values->end())
      reject("training component enum field values must be unique");
  } else if (field.type == TrainingValueType::enumeration) {
    reject("training component enum field requires declared values");
  }
  if ((field.type == TrainingValueType::string_list) !=
      field.collection_semantics.has_value())
    reject("training component string-list fields require collection semantics");
  if (field.string_format && field.type != TrainingValueType::string &&
      field.type != TrainingValueType::string_list)
    reject("training component string format requires a string field");
  if (state_field && (field.default_value || field.minimum || field.maximum ||
                      field.values || field.unit))
    reject("training component state fields describe shape, not configuration");
  if (field.default_value) {
    if (!value_has_type(field.type, *field.default_value) ||
        !bounded_scalar(*field.default_value))
      reject("training component field default has the wrong type");
    canonicalize_collection(field, *field.default_value);
    if (field.string_format &&
        !formatted_string(*field.string_format, *field.default_value))
      reject("training component field default violates its string format");
    if (field.type == TrainingValueType::path) {
      const std::filesystem::path path =
          field.default_value->get_ref<const std::string&>();
      if (!path.is_absolute() || !std::filesystem::exists(path))
        reject("training component path default is unavailable");
    }
    const double numeric = field.default_value->is_number()
                               ? field.default_value->get<double>()
                               : 0.0;
    if ((field.minimum && numeric < *field.minimum) ||
        (field.maximum && numeric > *field.maximum) ||
        (field.values &&
         std::ranges::find(*field.values, *field.default_value) ==
             field.values->end()))
      reject("training component field default violates its contract");
  }
}

void canonical_fields(std::vector<TrainingComponentField>& fields,
                      bool state_fields) {
  if (fields.size() > kMaximumFields)
    reject("training component field list exceeds its bound");
  for (TrainingComponentField& field : fields)
    validate_field(field, state_fields);
  std::ranges::sort(fields, {}, &TrainingComponentField::name);
  if (std::ranges::adjacent_find(
          fields, {}, &TrainingComponentField::name) != fields.end())
    reject("training component field names must be unique");
}

void validate_descriptor(TrainingComponentDescriptor& descriptor) {
  if (!symbolic_identity(descriptor.key.name) ||
      !symbolic_identity(descriptor.key.version, false, true) ||
      !symbolic_identity(descriptor.implementation))
    reject("training component key or implementation is malformed");
  descriptor.model_families = canonical_strings(
      std::move(descriptor.model_families), "model_families", true);
  if (std::ranges::contains(descriptor.model_families, std::string{"*"}) &&
      descriptor.model_families.size() != 1U)
    reject("wildcard model-family compatibility must be declared alone");
  if (!descriptor.required_capabilities.empty()) {
    descriptor.required_capabilities = canonical_strings(
        std::move(descriptor.required_capabilities),
        "required_capabilities");
  }
  canonical_fields(descriptor.configuration, false);
  canonical_fields(descriptor.state, true);
  if (scheduled(descriptor.key.category) != descriptor.step_domain.has_value())
    reject("schedule-like training components require exactly one step domain");
  if ((descriptor.state_grade == TrainingStateGrade::stateless) !=
      descriptor.state.empty())
    reject("training component state grade disagrees with its state schema");
  if (descriptor.reference_implementation &&
      descriptor.backend == TrainingComponentBackend::cuda_extension)
    reject("a CUDA extension cannot be the portable reference implementation");
}

Json resolve_configuration(const TrainingComponentDescriptor& descriptor,
                           const Json& requested) {
  if (!requested.is_object() ||
      requested.dump().size() > kMaximumConfigurationBytes)
    reject("training component configuration must be an object");
  Json resolved = Json::object();
  std::map<std::string, const TrainingComponentField*, std::less<>> fields;
  for (const TrainingComponentField& field : descriptor.configuration) {
    fields.emplace(field.name, &field);
    if (field.default_value) resolved[field.name] = *field.default_value;
  }
  for (const auto& [name, value] : requested.items()) {
    const auto field = fields.find(name);
    if (field == fields.end())
      reject("training component configuration contains an unknown field");
    const TrainingComponentField& contract = *field->second;
    if (!value_has_type(contract.type, value) || !bounded_scalar(value))
      reject("training component configuration field has the wrong type");
    Json canonical = value;
    canonicalize_collection(contract, canonical);
    if (contract.string_format &&
        !formatted_string(*contract.string_format, canonical))
      reject("training component configuration violates a string format");
    if (contract.type == TrainingValueType::path) {
      const std::filesystem::path path =
          canonical.get_ref<const std::string&>();
      if (!path.is_absolute() || !std::filesystem::exists(path))
        reject("training component configuration path is unavailable");
    }
    if (value.is_number()) {
      const double numeric = value.get<double>();
      if ((contract.minimum && numeric < *contract.minimum) ||
          (contract.maximum && numeric > *contract.maximum))
        reject("training component configuration violates a numeric bound");
    }
    if (contract.values &&
        std::ranges::find(*contract.values, value) == contract.values->end())
      reject("training component configuration violates an enum contract");
    resolved[name] = std::move(canonical);
  }
  for (const TrainingComponentField& field : descriptor.configuration) {
    if (field.required && !resolved.contains(field.name))
      reject("training component configuration is missing a required field");
  }
  return resolved;
}

Json canonical_descriptor(const TrainingComponentDescriptor& descriptor) {
  return encode_json(descriptor);
}

std::map<TrainingComponentCategory,
         std::vector<const ResolvedTrainingComponent*>>
components_by_category(const ResolvedTrainingComposition& composition) {
  std::map<TrainingComponentCategory,
           std::vector<const ResolvedTrainingComponent*>> grouped;
  for (const auto& [slot, component] : composition.components) {
    (void)slot;
    grouped[component.descriptor.key.category].push_back(&component);
  }
  return grouped;
}

void validate_model_trainability_relationships(
    const ResolvedTrainingComposition& composition) {
  const auto grouped = components_by_category(composition);
  const auto loaders = grouped.find(TrainingComponentCategory::model_loader);
  const auto policies = grouped.find(TrainingComponentCategory::trainability);
  const std::size_t loader_count =
      loaders == grouped.end() ? 0U : loaders->second.size();
  const std::size_t policy_count =
      policies == grouped.end() ? 0U : policies->second.size();
  if (loader_count > 1U || policy_count > 1U)
    reject("training composition may select only one model loader and one trainability policy");
  if ((loader_count == 0U) != (policy_count == 0U))
    reject("training composition must select model loader and trainability together");
  if (loader_count == 0U) return;

  const auto& loader = *loaders->second.front();
  const auto& policy = *policies->second.front();
  const std::string quantization =
      loader.configuration.value("quantization", std::string{"none"});
  if (quantization != "none" &&
      policy.descriptor.implementation ==
          "rwkv_lab.trainability.full.v1")
    reject("full trainability is incompatible with a quantized model loader");
}

void validate_data_pipeline_relationships(
    const ResolvedTrainingComposition& composition) {
  const auto grouped = components_by_category(composition);
  constexpr std::array categories{
      TrainingComponentCategory::data_source,
      TrainingComponentCategory::sample_processor,
      TrainingComponentCategory::sample_mapper,
      TrainingComponentCategory::collator,
      TrainingComponentCategory::sampler,
      TrainingComponentCategory::batching,
      TrainingComponentCategory::split_selector,
  };
  std::size_t selected = 0U;
  for (const auto category : categories) {
    const auto components = grouped.find(category);
    const std::size_t count =
        components == grouped.end() ? 0U : components->second.size();
    if (count > 1U)
      reject("training composition may select only one component per data-pipeline category");
    selected += count;
  }
  if (selected == 0U) return;
  if (selected != categories.size())
    reject("training composition must select the complete declarative data pipeline");

  const auto& source =
      *grouped.at(TrainingComponentCategory::data_source).front();
  const auto& processor =
      *grouped.at(TrainingComponentCategory::sample_processor).front();
  const auto& mapper =
      *grouped.at(TrainingComponentCategory::sample_mapper).front();
  const auto& batching =
      *grouped.at(TrainingComponentCategory::batching).front();

  const bool image_source = source.descriptor.implementation ==
                            "rwkv_lab.data_source.jsonl_image_caption.v1";
  const bool image_processor = processor.descriptor.implementation ==
                               "rwkv_lab.sample_processor.image_caption.v1";
  const bool assistant_mapper = mapper.descriptor.implementation ==
                                "rwkv_lab.sample_mapper.assistant_only.v1";
  if (image_source != image_processor || image_processor != assistant_mapper)
    reject("training data source, processor and mapper modalities are incompatible");
  if (batching.configuration.value("bucket_by", std::string{"token_length"}) ==
          "image_area" &&
      !image_source)
    reject("image-area bucketing requires an image-caption data source");

  const auto declared =
      source.configuration.at("declared_columns").get<std::vector<std::string>>();
  const auto has_column = [&declared](const std::string& column) {
    return column.empty() || std::ranges::contains(declared, column);
  };
  std::vector<std::string> referenced;
  if (image_source) {
    referenced.push_back(source.configuration.at("image_column"));
    for (const auto& column : source.configuration.at("caption_columns"))
      referenced.push_back(column.get<std::string>());
    referenced.push_back(processor.configuration.at("image_column"));
    for (const auto& column : processor.configuration.at("caption_columns"))
      referenced.push_back(column.get<std::string>());
    referenced.push_back(mapper.configuration.at("prompt_column"));
    referenced.push_back(mapper.configuration.at("target_column"));
    const bool has_prompt_column =
        !mapper.configuration.at("prompt_column").get<std::string>().empty();
    const bool has_fixed_prompt =
        !mapper.configuration.at("fixed_prompt").get<std::string>().empty();
    if (has_prompt_column == has_fixed_prompt)
      reject("assistant-only mapping must select exactly one prompt source");
  } else {
    referenced.push_back(source.configuration.at("token_column"));
    referenced.push_back(processor.configuration.at("token_column"));
    referenced.push_back(mapper.configuration.at("token_column"));
  }
  referenced.push_back(source.configuration.at("id_column"));
  if (!std::ranges::all_of(referenced, has_column))
    reject("data pipeline references a column absent from declared_columns");
}

Json composition_body(const ResolvedTrainingComposition& composition) {
  Json components = Json::object();
  for (const auto& [slot, component] : composition.components) {
    components[slot] = {
        {"configuration", component.configuration},
        {"descriptor", canonical_descriptor(component.descriptor)},
        {"descriptor_digest", component.descriptor_digest},
    };
  }
  Json body{{"api_version", "trainvm.resolved-training-composition/v1"},
            {"components", std::move(components)},
            {"model_family", composition.model_family},
            {"registry_digest", composition.registry_digest}};
  if (!composition.topologies.is_null())
    body["topologies"] = composition.topologies;
  if (!composition.post_training.is_null())
    body["post_training"] = composition.post_training;
  return body;
}

}  // namespace

TrainingComponentRegistry::TrainingComponentRegistry(
    std::vector<TrainingComponentDescriptor> descriptors) {
  if (descriptors.size() > kMaximumComponents)
    reject("training component registry exceeds its component bound");
  for (TrainingComponentDescriptor& descriptor : descriptors) {
    validate_descriptor(descriptor);
    const TrainingComponentKey key = descriptor.key;
    if (!descriptors_.emplace(key, std::move(descriptor)).second)
      reject("training component registry contains a duplicate exact key");
  }
  const std::string canonical_registry =
      Json{{"api_version", "trainvm.training-components/v1"},
           {"components", descriptors_json()}}
          .dump();
  if (canonical_registry.size() > kMaximumRegistryBytes)
    reject("training component registry exceeds its canonical byte bound");
  registry_digest_ = "sha256:" + sha256_hex(canonical_registry);
}

TrainingComponentRegistry TrainingComponentRegistry::from_json(
    std::string_view document) {
  if (document.empty() || document.size() > kMaximumRegistryBytes)
    reject("training component registry document size is invalid");
  Json source;
  bool duplicate_key = false;
  std::vector<std::set<std::string>> object_keys;
  try {
    const Json::parser_callback_t reject_duplicates =
        [&](int depth, Json::parse_event_t event, Json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == Json::parse_event_t::object_start) {
            if (object_keys.size() <= index + 1U)
              object_keys.resize(index + 2U);
            object_keys[index + 1U].clear();
          } else if (event == Json::parse_event_t::key) {
            if (object_keys.size() <= index) object_keys.resize(index + 1U);
            if (!object_keys[index].insert(parsed.get<std::string>()).second)
              duplicate_key = true;
          } else if (event == Json::parse_event_t::object_end &&
                     object_keys.size() > index + 1U) {
            object_keys[index + 1U].clear();
          }
          return true;
        };
    source = Json::parse(document, reject_duplicates);
  } catch (const Json::exception& exception) {
    reject("training component registry is not valid JSON: " +
           std::string(exception.what()));
  }
  if (duplicate_key)
    reject("training component registry contains a duplicate object key");
  TrainingComponentRegistryDocument decoded;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, decoded, "", diagnostics))
    reject("training component registry schema validation failed: " +
           diagnostics_json(diagnostics).dump());
  if (decoded.api_version != "trainvm.training-components/v1")
    reject("training component registry api_version is unsupported");
  return TrainingComponentRegistry(std::move(decoded.components));
}

TrainingComponentRegistry TrainingComponentRegistry::load_file(
    const std::filesystem::path& path) {
  return from_json(read_authority_document(
      path, "training component registry", kMaximumRegistryBytes));
}

const TrainingComponentDescriptor& TrainingComponentRegistry::descriptor(
    const TrainingComponentKey& key) const {
  const auto found = descriptors_.find(key);
  if (found == descriptors_.end())
    reject("no training component matches the exact requested key");
  return found->second;
}

ResolvedTrainingComponent TrainingComponentRegistry::resolve(
    const TrainingComponentRequest& request) const {
  if (!symbolic_identity(request.model_family))
    reject("training component request has an invalid model family");
  const TrainingComponentDescriptor& selected = descriptor(request.key);
  if (!std::ranges::contains(selected.model_families, std::string{"*"}) &&
      !std::ranges::contains(selected.model_families, request.model_family))
    reject("training component is incompatible with the requested model family");
  return {
      .descriptor = selected,
      .configuration =
          resolve_configuration(selected, request.configuration),
      .descriptor_digest = descriptor_digest(request.key),
  };
}

ResolvedTrainingComposition TrainingComponentRegistry::resolve_composition(
    const TrainingComposition& composition) const {
  if (!symbolic_identity(composition.model_family) ||
      composition.components.empty() ||
      composition.components.size() > 64U)
    reject("training composition identity or component count is invalid");
  ResolvedTrainingComposition resolved{
      .model_family = composition.model_family,
      .components = {},
      .topologies = nullptr,
      .post_training = nullptr,
      .registry_digest = registry_digest_,
      .composition_digest = {},
  };
  if (composition.topologies) {
    std::vector<RwkvScratchSelection> selections;
    for (const TrainingTopologySelection& chosen : *composition.topologies) {
      const auto topology = rwkv_scratch_topology_from_name(chosen.topology);
      if (!topology) reject("training composition names an unknown topology");
      std::map<std::string, nlohmann::json> assignments;
      for (const auto& [name, value] : chosen.parameters.items())
        assignments.emplace(name, value);
      selections.push_back(
          {.topology = *topology, .assignments = std::move(assignments)});
    }
    // Throws on an undeclared switch, a bound violation, a duplicate, or a
    // declared-incompatible pair. Compile already checked; this is the
    // authority-side repeat so a plan cannot reach a worker unvalidated.
    resolved.topologies = rwkv_scratch_training_block(selections);
  }
  if (composition.post_training) {
    // Same discipline as the topology block above: compile already validated
    // this, and the authority repeats it so a plan cannot reach a worker
    // unvalidated. Refused here rather than lowered silently, because the
    // worker has no way to tell a missing arm from a rejected one.
    const PostTrainingArmLowering lowering =
        lower_post_training_arm(*composition.post_training);
    if (!lowering.complete())
      reject("training composition post-training arm names an unknown kind, "
             "claim or bound");
    if (const auto refusal =
            validate_post_training_arm_declaration(lowering.arm))
      reject("training composition post-training arm is invalid: " +
             refusal->message);
    resolved.post_training = post_training_arm_json(lowering.arm);
  }
  for (const auto& [slot, selection] : composition.components) {
    if (!symbolic_identity(slot))
      reject("training composition slot identity is invalid");
    resolved.components.emplace(
        slot, resolve({.key = selection.key,
                       .model_family = composition.model_family,
                       .configuration = selection.configuration}));
  }
  validate_model_trainability_relationships(resolved);
  validate_data_pipeline_relationships(resolved);
  const std::string canonical_composition = composition_body(resolved).dump();
  if (canonical_composition.size() > kMaximumCompositionBytes)
    reject("resolved training composition exceeds its canonical byte bound");
  resolved.composition_digest =
      "sha256:" + sha256_hex(canonical_composition);
  return resolved;
}

void TrainingComponentRegistry::validate_resume_state(
    const ResolvedTrainingComposition& composition, const Json& state) const {
  if (!state.is_object() || state.dump().size() > kMaximumCompositionBytes)
    reject("training component resume state must be a bounded object");

  std::map<std::string, const ResolvedTrainingComponent*, std::less<>>
      stateful;
  for (const auto& [slot, component] : composition.components) {
    if (component.descriptor.state_grade != TrainingStateGrade::stateless)
      stateful.emplace(slot, &component);
  }
  if (state.size() != stateful.size())
    reject("training component resume state has incomplete slot coverage");
  for (const auto& [slot, value] : state.items()) {
    const auto selected = stateful.find(slot);
    if (selected == stateful.end() || !value.is_object())
      reject("training component resume state contains an unknown slot");
    const auto& fields = selected->second->descriptor.state;
    if (value.size() != fields.size())
      reject("training component resume state has incomplete field coverage");
    for (const TrainingComponentField& field : fields) {
      if (!value.contains(field.name))
        reject("training component resume state is missing a required field");
      Json item = value.at(field.name);
      if (!value_has_type(field.type, item) || !bounded_scalar(item))
        reject("training component resume state field has the wrong type");
      canonicalize_collection(field, item);
      if (item != value.at(field.name))
        reject("training component resume state is not canonical");
      if (field.string_format &&
          !formatted_string(*field.string_format, item))
        reject("training component resume state violates a string format");
    }
  }
}

WorkerLaunchRequest TrainingComponentRegistry::augment_worker_launch_request(
    WorkerLaunchRequest request,
    const std::optional<TrainingComposition>& composition) const {
  if (!composition) return request;
  const ResolvedTrainingComposition resolved =
      resolve_composition(*composition);
  for (const auto& [slot, component] : resolved.components) {
    (void)slot;
    request.required_capabilities.insert(
        request.required_capabilities.end(),
        component.descriptor.required_capabilities.begin(),
        component.descriptor.required_capabilities.end());
  }
  std::ranges::sort(request.required_capabilities);
  request.required_capabilities.erase(
      std::ranges::unique(request.required_capabilities).begin(),
      request.required_capabilities.end());
  if (request.required_capabilities.size() > kMaximumValues)
    reject("composed worker capabilities exceed their bound");
  return request;
}

Json resolved_training_composition_json(
    const ResolvedTrainingComposition& composition) {
  if (composition.composition_digest !=
      "sha256:" + sha256_hex(composition_body(composition).dump()))
    reject("resolved training composition digest is not canonical");
  Json result = composition_body(composition);
  result["composition_digest"] = composition.composition_digest;
  return result;
}

bool TrainingComponentRegistry::plan_uses_components(
    const CompiledPlan& plan) const {
  return std::ranges::any_of(
      plan.experiment.spec.workflow.nodes, [](const auto& item) {
        return item.second.invoke.training.has_value();
      });
}

void TrainingComponentRegistry::validate_plan(const CompiledPlan& plan) const {
  for (const auto& [node_name, node] :
       plan.experiment.spec.workflow.nodes) {
    if (!node.invoke.training) continue;
    const Component& component =
        plan.experiment.spec.components.at(node.invoke.component);
    if (component.runtime == ComponentRuntime::builtin ||
        node.effect != Effect::process)
      reject("workflow node " + node_name +
             " attaches training components to a non-worker process operation");
    const ResolvedTrainingComposition resolved =
        resolve_composition(*node.invoke.training);
    if (plan.experiment.spec.recovery.exact_resume &&
        std::ranges::any_of(
            resolved.components, [](const auto& item) {
              return item.second.descriptor.state_grade ==
                     TrainingStateGrade::compatible;
            }))
      reject("workflow node " + node_name +
             " requests exact resume with a compatibility-grade training component");
  }
}

std::string TrainingComponentRegistry::plan_lock_manifest(
    const CompiledPlan& plan) const {
  validate_plan(plan);
  Json nodes = Json::object();
  for (const auto& [node_name, node] :
       plan.experiment.spec.workflow.nodes) {
    if (node.invoke.training) {
      nodes[node_name] = resolved_training_composition_json(
          resolve_composition(*node.invoke.training));
    }
  }
  return Json{{"api_version", "trainvm.training-component-lock/v1"},
              {"nodes", std::move(nodes)},
              {"registry_digest", registry_digest_}}
      .dump();
}

std::string TrainingComponentRegistry::plan_lock_digest(
    const CompiledPlan& plan) const {
  return "sha256:" + sha256_hex(plan_lock_manifest(plan));
}

void TrainingComponentRegistry::validate_submission_lock(
    const CompiledPlan& plan, const Json& submission) const {
  if (!plan_uses_components(plan)) return;
  const std::string manifest = plan_lock_manifest(plan);
  const std::string digest = "sha256:" + sha256_hex(manifest);
  if (!submission.is_object() ||
      submission.value("training_component_lock_digest", std::string{}) !=
          digest ||
      !submission.contains("training_component_lock") ||
      submission.at("training_component_lock") != Json::parse(manifest))
    reject("run training-component lock differs from the authority registry");
}

const std::string& TrainingComponentRegistry::registry_digest() const noexcept {
  return registry_digest_;
}

std::string TrainingComponentRegistry::descriptor_digest(
    const TrainingComponentKey& key) const {
  return "sha256:" + sha256_hex(canonical_descriptor(descriptor(key)).dump());
}

Json TrainingComponentRegistry::descriptors_json() const {
  Json components = Json::array();
  for (const auto& [key, descriptor] : descriptors_) {
    (void)key;
    components.push_back(canonical_descriptor(descriptor));
  }
  return components;
}

Json TrainingComponentRegistry::document_json() const {
  Json document{{"api_version", "trainvm.training-components/v1"},
                {"components", descriptors_json()}};
  if ("sha256:" + sha256_hex(document.dump()) != registry_digest_)
    reject("training component registry document identity is not canonical");
  return document;
}

}  // namespace trainvm
