from owlready2 import get_ontology
import json
from tqdm import tqdm

# Load the ontology
fma = get_ontology("http://purl.org/sig/ont/fma.owl").load()

# Set the maximum depth for the hierarchy
MAX_DEPTH = 5

def get_entity_info(entity):
    info = {
        "name": get_name(entity),
        "iri": entity.iri,
        "comment": entity.comment[0] if entity.comment else "",
        "synonyms": list(entity.synonym) if hasattr(entity, 'synonym') else [],
        "radlex_id": entity.RadLex_ID[0] if (hasattr(entity, 'RadLex_ID') and len(entity.RadLex_ID) > 0) else "",
        "relations": {
            "constitutional_part": [get_name(part) for part in entity.constitutional_part],
            "regional_part": [get_name(part) for part in entity.regional_part],
            "branch": [get_name(branch) for branch in entity.branch if isinstance(branch, entity.__class__)],
            "is_a": [get_name(parent) for parent in entity.is_a if isinstance(parent, entity.__class__)],
            "constitutional_part_of": [get_name(whole) for whole in entity.constitutional_part_of],
            "regional_part_of": [get_name(whole) for whole in entity.regional_part_of],
            "branch_of": [get_name(trunk) for trunk in entity.branch_of if isinstance(trunk, entity.__class__)]
        }
    }
    return info

def get_name(entity):
    if hasattr(entity, 'preferred_name') and entity.preferred_name:
        return entity.preferred_name[0]
    elif hasattr(entity, 'name'):
        return entity.name
    else:
        return str(entity)

def build_hierarchy(entity, visited=None, current_depth=0):
    if visited is None:
        visited = set()

    if entity in visited or current_depth >= MAX_DEPTH:
        return {}

    visited.add(entity)
    entity_info = get_entity_info(entity)
    
    hierarchy = {entity_info["name"]: {"info": entity_info, "children": {}}}

    if current_depth < MAX_DEPTH:
        relations = [
            entity.constitutional_part,
            entity.regional_part,
            entity.branch,
            entity.subclasses(),
        ]

        for related_entities in relations:
            for related_entity in related_entities:
                if related_entity not in visited:
                    child_hierarchy = build_hierarchy(related_entity, visited, current_depth + 1)
                    if child_hierarchy:
                        hierarchy[entity_info["name"]]["children"].update(child_hierarchy)

    return hierarchy

# List of relevant nodes
nodes = [
    "Head",
    "Set of subarachnoid spaces",
    "Set of subdivisions of head",
    "Subdivision of head",
    "Subdivision of head proper",
    "Space of compartment of head",
    "Epidural space",
    "Set of subarachnoid cisterns",
    "Set of white matter structures of neuraxis",
    "Set of gray matter structures of neuraxis",
    "Common carotid artery",
    "Vertebral artery",
    "Subdivision of brachiocephalic vein", # includes dural venous sinuses
    "Subdivision of face",
    "Set of eye and related structures",
    "Set of accessory visual structures",
    "Set of paranasal sinuses",
    "Nasal skeleton",
]

# Build the hierarchy starting from each node
full_hierarchy = {}
for node_name in tqdm(nodes, desc="Processing nodes"):
    node = fma.search_one(preferred_name=node_name)
    if node:
        hierarchy = build_hierarchy(node)
        full_hierarchy.update(hierarchy)
    else:
        print(f"Warning: '{node_name}' not found in the ontology")

# Define the output file name as a constant
OUTPUT_FILE = f"fma_head_hierarchy_maxdepth_{MAX_DEPTH}.json"

# Save the full hierarchy to JSON
print(f"Saving hierarchy to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as json_file:
    json.dump(full_hierarchy, json_file, indent=4)

print(f"Hierarchy saved to {OUTPUT_FILE}")

# Print some statistics
print(f"Number of initial nodes: {len(nodes)}")
print(f"Total number of nodes in hierarchy: {sum(1 for _ in str(full_hierarchy).split('{'))}")
