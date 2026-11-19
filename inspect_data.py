"""v2.4.0 corpus builder execution."""
import json, os, sys
sys.path.insert(0, '.')

# List data files
data_dir = os.path.join('.', 'data')
py_files = []
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.py'):
            py_files.append(os.path.join(root, f))

print('Python files in data/:')
for f in sorted(py_files):
    print(f'  {f}')

# Check for build scripts
print('\nLooking for build scripts...')
for f in py_files:
    name = os.path.basename(f)
    if 'build' in name.lower() or 'corpus' in name.lower() or 'instruct' in name.lower():
        print(f'  Found: {name}')