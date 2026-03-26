#!/usr/bin/env python3
"""
Project Digest Tool

Converts a project folder into a single markdown file optimized for LLM context.
Similar to AI Digest or Gitingest tools.

Usage:
    python project_digest.py /path/to/project -o output.md
    python project_digest.py /path/to/project  # outputs to project_digest.md
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

# File extensions to include (code and config files)
INCLUDE_EXTENSIONS = {
    '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.toml',
    '.sh', '.bash', '.cfg', '.ini', '.conf',
    '.html', '.css', '.js', '.jsx', '.ts', '.tsx',
}

# Directories to skip
SKIP_DIRS = {
    '__pycache__', '.git', '.svn', '.hg',
    'node_modules', '.venv', 'venv', 'env',
    '.pytest_cache', '.mypy_cache', '.tox',
    'dist', 'build', '*.egg-info',
    '.ipynb_checkpoints', '.idea', '.vscode',
    'results', 'outputs', 'logs', 'checkpoints',
    'wandb', 'mlruns', 'lightning_logs',
}

# Files to skip
SKIP_FILES = {
    '.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes',
    'package-lock.json', 'yarn.lock', 'poetry.lock',
}

# Max file size to include (in bytes) - skip very large files
MAX_FILE_SIZE = 100 * 1024  # 100 KB


def should_skip_dir(dirname):
    """Check if directory should be skipped."""
    return dirname in SKIP_DIRS or dirname.startswith('.')


def should_include_file(filepath):
    """Check if file should be included."""
    path = Path(filepath)
    
    # Skip by name
    if path.name in SKIP_FILES:
        return False
    
    # Skip hidden files
    if path.name.startswith('.'):
        return False
    
    # Skip by size
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    
    # Include by extension
    if path.suffix.lower() in INCLUDE_EXTENSIONS:
        return True
    
    # Include files without extension if they look like scripts
    if path.suffix == '':
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                if first_line.startswith('#!'):
                    return True
        except:
            pass
    
    return False


def get_language(filepath):
    """Get markdown language identifier for syntax highlighting."""
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'jsx',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown',
        '.sh': 'bash',
        '.bash': 'bash',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.txt': 'text',
    }
    ext = Path(filepath).suffix.lower()
    return ext_to_lang.get(ext, '')


def build_tree(root_path):
    """Build directory tree structure."""
    tree_lines = []
    root = Path(root_path)
    
    def add_to_tree(path, prefix=""):
        entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        
        # Filter entries
        dirs = [e for e in entries if e.is_dir() and not should_skip_dir(e.name)]
        files = [e for e in entries if e.is_file() and should_include_file(e)]
        
        all_entries = dirs + files
        
        for i, entry in enumerate(all_entries):
            is_last = (i == len(all_entries) - 1)
            connector = "└── " if is_last else "├── "
            
            if entry.is_dir():
                tree_lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if is_last else "│   "
                add_to_tree(entry, prefix + extension)
            else:
                tree_lines.append(f"{prefix}{connector}{entry.name}")
    
    tree_lines.append(f"{root.name}/")
    add_to_tree(root)
    
    return "\n".join(tree_lines)


def collect_files(root_path):
    """Collect all files to include."""
    files = []
    root = Path(root_path)
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out directories to skip
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if should_include_file(filepath):
                rel_path = os.path.relpath(filepath, root)
                files.append((rel_path, filepath))
    
    # Sort by path
    files.sort(key=lambda x: x[0].lower())
    return files


def read_file_content(filepath):
    """Read file content with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file: {e}]"


def generate_digest(root_path, project_name=None):
    """Generate the markdown digest."""
    root = Path(root_path)
    
    if project_name is None:
        project_name = root.name
    
    # Header
    lines = [
        f"# {project_name} - Project Digest",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This document contains the complete source code and structure of the project,",
        "formatted for LLM context windows.",
        "",
        "---",
        "",
    ]
    
    # Table of Contents
    lines.extend([
        "## Table of Contents",
        "",
        "1. [Project Structure](#project-structure)",
        "2. [Source Files](#source-files)",
        "",
        "---",
        "",
    ])
    
    # Project Structure
    lines.extend([
        "## Project Structure",
        "",
        "```",
        build_tree(root_path),
        "```",
        "",
        "---",
        "",
    ])
    
    # Source Files
    lines.extend([
        "## Source Files",
        "",
    ])
    
    files = collect_files(root_path)
    
    for rel_path, filepath in files:
        lang = get_language(filepath)
        content = read_file_content(filepath)
        
        # File header
        lines.extend([
            f"### `{rel_path}`",
            "",
            f"```{lang}",
            content,
            "```",
            "",
        ])
    
    # Footer with stats
    total_files = len(files)
    total_lines = sum(len(read_file_content(fp).splitlines()) for _, fp in files)
    
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        f"- **Total files:** {total_files}",
        f"- **Total lines:** {total_lines:,}",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a markdown digest of a project for LLM context'
    )
    parser.add_argument(
        'project_path',
        type=str,
        help='Path to the project directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: project_digest.md)'
    )
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Project name for the header'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=100,
        help='Max file size in KB to include (default: 100)'
    )
    
    args = parser.parse_args()
    
    global MAX_FILE_SIZE
    MAX_FILE_SIZE = args.max_size * 1024
    
    project_path = os.path.abspath(args.project_path)
    
    if not os.path.isdir(project_path):
        print(f"Error: {project_path} is not a directory")
        return 1
    
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.getcwd(), 'project_digest.md')
    
    print(f"Generating digest for: {project_path}")
    print(f"Output: {output_path}")
    
    digest = generate_digest(project_path, args.name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(digest)
    
    # Print stats
    files = collect_files(project_path)
    print(f"\nDone! Included {len(files)} files")
    print(f"Output size: {len(digest):,} characters")
    
    return 0


if __name__ == '__main__':
    exit(main())