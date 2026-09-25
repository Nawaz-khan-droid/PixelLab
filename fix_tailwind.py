import os
import re

replacements = {
    r'\bbg-surface-900\b': 'bg-white dark:bg-surface-900',
    r'\bbg-surface-800\b': 'bg-surface-50 dark:bg-surface-800',
    r'\bbg-surface-700\b': 'bg-surface-200 dark:bg-surface-700',
    r'\bborder-surface-700\b': 'border-surface-200 dark:border-surface-700',
    r'\bborder-surface-600\b': 'border-surface-300 dark:border-surface-600',
    r'\btext-surface-400\b': 'text-surface-600 dark:text-surface-400',
    r'\btext-surface-300\b': 'text-surface-800 dark:text-surface-300',
    r'\bhover:bg-surface-800\b': 'hover:bg-surface-100 dark:hover:bg-surface-800',
    r'\bhover:bg-surface-700\b': 'hover:bg-surface-200 dark:hover:bg-surface-700',
    r'\bhover:text-surface-400\b': 'hover:text-surface-600 dark:hover:text-surface-400',
    r'\bhover:text-surface-300\b': 'hover:text-surface-800 dark:hover:text-surface-300',
}

# Ensure we don't double replace
def safe_replace(content):
    # First, temporarily mask already correct ones if they exist (unlikely but safe)
    # Actually, a simple pass is fine if we just run it once.
    for k, v in replacements.items():
        # Only replace if not preceded by dark: and not already part of the replacement
        content = re.sub(r'(?<!dark:)' + k, v, content)
    return content

for root, _, files in os.walk('templates'):
    for f in files:
        if f.endswith('.html'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            new_content = safe_replace(content)
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"Updated {path}")
