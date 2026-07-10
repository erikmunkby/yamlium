# Yamlium

A high-performance, dependency-free YAML parser for Python that preserves all YAML features including comments, anchors, and formatting.

## 📦 Features

- 🎯 **First-Class YAML Features**: Preserves all YAML elements including comments, newlines, anchor names, and formatting
- ⚡ **High Performance**: 3x faster than [PyYAML](https://pypi.org/project/PyYAML/)
- 🧹 **Zero Dependencies**: Pure Python implementation with no external dependencies
- 🛡️ **Type Safety**: Full type hints support
- 🛠️ **Rich API**: Intuitive interface for manipulating YAML structures

## 🛠️ Installation

```bash
pip install yamlium
```

## 🚀 Quick Start

### Basic Parsing

```python
from yamlium import parse

# Parse a YAML string
yaml_str = """
name: John Doe
age: 30
address:
  street: 123 Main St
  city: Boston
"""
data = parse(yaml_str)

# Access values
print(data["name"])  # John Doe
print(data["address"]["city"])  # Boston
```

### Preserving YAML Features

```python
from yamlium import parse

yaml_str = """
# User configuration
user: &user_ref # Anchor definition
  name: Alice
  role: admin

# Reference to user
admin: *user_ref # Alias reference

""".lstrip()
yml = parse(yaml_str)

# The YAML structure is preserved when converting back including:
# - Anchor names
# - Comments
# - Newlines between objects
print(yml.to_yaml() == yaml_str)
```

### Manipulating YAML

```python
from yamlium import parse

yaml_str = """
users: # List of users
  - name: alice
    age: 25
  - name: Bob
    age: 30
  - name: charlie
"""
yml = parse(yaml_str)

# Modify values while preserving structure
for key, value, obj in yml.walk_keys():
    if key == "age":
        value += 1
    elif key == "name":
        # Using the string manipulation interface `.str`
        obj[key] = value.str.capitalize()

print(yml.to_yaml())
```

### Working with Comments

Yamlium provides structured access to YAML comments via the `comments` attribute:

```python
from yamlium import parse

yaml_str = """
app:
  # Database settings
  db_host: localhost # primary host
  # TODO: increase timeout

  # Cache config
  cache_ttl: 3600
"""
yml = parse(yaml_str)

# Access comments on a key
db_key = list(yml["app"].keys())[0]
print(db_key.comments.head)  # ['# Database settings']

# Access inline and foot comments on a value
db_value = yml["app"]["db_host"]
print(db_value.comments.line)  # '# primary host'
print(db_value.comments.foot)  # ['# TODO: increase timeout']

# Modify comments
db_value.comments.line = "# updated comment"
```

Comment types:
- `comments.head` - comments directly above a node
- `comments.line` - inline comment on the same line
- `comments.foot` - comments below a node (before a blank line)

### Reading Markdown Frontmatter

```python
from yamlium import read_markdown

# Parse a markdown file with YAML frontmatter
frontmatter, content = read_markdown("post.md")

if frontmatter:
    print(frontmatter["title"])  # Access YAML fields
    print(content)               # Raw markdown after the frontmatter

# Also works with raw strings
text = """---
title: My Post
tags: [python, yaml]
---
# Hello World

This is the body.
"""
frontmatter, content = read_markdown(text)
```

Supports both standard (`---`-delimited) and open (bare YAML followed by `---`) frontmatter formats.
If no frontmatter is detected, returns `(None, full_text)`.

### JSON Conversion

```python
from yamlium import from_json, from_dict

# Convert from JSON string
json_str = '{"name": "test", "values": [1, 2, 3]}'
yaml_data = from_json(json_str)

# Convert from Python dict
python_dict = {"name": "test", "values": [1, 2, 3]}
yaml_data = from_dict(python_dict)
```

## 📚 API Reference

### Parsing Functions

- `parse(input: str | Path) -> Mapping` Parse a single YAML document
- `parse_full(input: str | Path) -> Document` Parse multiple YAML documents
- `read_markdown(input: str | Path) -> tuple[Mapping | None, str]` Extract YAML frontmatter and content from markdown
- `from_json(input: str | Path) -> Mapping | Sequence` Convert JSON to YAML structure
- `from_dict(input: dict | list) -> Mapping | Sequence` Convert Python dict/list to YAML structure

### Yaml object functions
Given:
```py
from yamlium import parse
yml = parse("my_yaml.yml")
```
- `yml.to_yaml()` Convert to yaml string
- `yml.to_dict()` Convert to python dictionary
- `yml.yaml_dump(destination="my_yaml.yml")` Write directly to yaml file
- `yml.pprint()` Pretty print the dictionary
- `yml.walk()` Iterate through all yaml objects
- `yml.walk_keys()` Iterate through all yaml keys


## 🔄 Comparison to PyYaml
While [PyYaml](https://pypi.org/project/PyYAML/) solves the purpose of converting to dictionary perfectly fine,
it completely ignores anything non-dictionary-conversion related in the yaml file.

### Input yaml
```yml
# Anchor definition
dev: &default_config
  schedule: false
  my_config: [1, 2, 3]

staging:
  # Alias reference
  <<: *default_config
  schedule: true
```
### Output
<table>
<tr>
<th> <code>yamlium</code> </th>
<th> <code>PyYaml</code> </th>
</tr>
<tr><td>✅ Retaining structure</td><td>❌ Changing structure</td></tr>
<tr>
<td>

```yml
# Anchor definition
dev: &default_config
  schedule: false
  my_config: [1, 2, 3]

staging:
  # Alias reference
  <<: *default_config
  schedule: true
```

</td>
<td>

```yml
dev:
  my_config: &id001
  - 1
  - 2
  - 3
  schedule: false
staging:
  my_config: *id001
  schedule: true
```
</td>
</tr>
</table>


## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues, Feature requests or Pull requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 