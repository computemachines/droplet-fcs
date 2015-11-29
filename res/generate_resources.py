#! /usr/bin/python3

"""Generate OpenCL

Usage:
  generate_resources.py <infile> -o <outfile>
"""

from docopt import docopt
import re
from os import path

def find_imports(lines):
    matches = {}
    for n in range(len(lines)):
        match = re.search(r'\/\/\s+import\s+([\w\.\/]+)', lines[n])
        if match:
            matches[n] = match.group(1)
    return matches

def replace_import_with_file(lines, _import=None):
    if _import==None:
        return replace_import_with_file(lines, _import=list(find_imports(lines).items())[0])
    else:
        (n, filename) = _import
        print('replacing import ({}, {})'.format(n, filename))
        lines.pop(n)
        lines.insert(n, '// endImport {}\n'.format(filename))
#        import ipdb; ipdb.set_trace()
        for importedLine in reversed(open(filename).readlines()):
            lines.insert(n, importedLine)
        lines.insert(n, '// beginImport {}\n'.format(filename))
        return lines

def process_lines(lines):
    """haskellish version
process_lines lines = 
    | lines.imports == 0 = lines
    | lines.imports > 0 = process_lines(replace_import_with_file(lines))
"""
    imports = find_imports(lines)
    if imports == {}:
        return lines
    else:
        replace_import_with_file(lines, _import=list(find_imports(lines).items())[0])
        return process_lines(lines)
        # process until imports == {}

if __name__ == '__main__':
    arguments = docopt(__doc__, version='v0.0')

    lines = open(arguments['<infile>']).readlines()
    out_lines = process_lines(lines)
    out_file = open(arguments['<outfile>'], 'w')
    out_file.write("".join(out_lines))
