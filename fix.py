#!/usr/bin/env python3
"""
Automated Fix for strategies/base.py
====================================

This script automatically fixes the critical indentation bugs in strategies/base.py
that cause the error: 'TrendFollowingStrategy' object has no attribute '_update_cooldowns'

Run from your project root: python fix_strategies_base.py

Author: AlphaTrade Fix Tool
"""

import re
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("AUTOMATED FIX FOR strategies/base.py")
    print("=" * 70)
    
    # Find the file
    base_path = Path("strategies/base.py")
    
    if not base_path.exists():
        print(f"\n❌ ERROR: {base_path} not found!")
        print("   Make sure you're running this from your project root directory.")
        return 1
    
    print(f"\n📂 Found: {base_path}")
    
    # Read the file
    content = base_path.read_text(encoding="utf-8")
    original_content = content
    
    # Create backup
    backup_path = Path("strategies/base.py.backup")
    backup_path.write_text(content, encoding="utf-8")
    print(f"💾 Backup created: {backup_path}")
    
    fixes_applied = []
    
    # =========================================================================
    # FIX 1: Fix _update_cooldowns method
    # =========================================================================
    print("\n🔍 Checking _update_cooldowns method...")
    
    # Pattern for incomplete _update_cooldowns (docstring followed by something other than 'for')
    # We need to find where _update_cooldowns ends and what comes after
    
    pattern1_incomplete = re.compile(
        r'(    def _update_cooldowns\(self\) -> None:\s*'
        r'"""Decrement cooldown counters\.""")'
        r'(\s*(?!        for))',
        re.MULTILINE
    )
    
    # Check if the for loop exists after the docstring
    check_pattern = re.compile(
        r'def _update_cooldowns\(self\) -> None:\s*'
        r'"""Decrement cooldown counters\."""\s*'
        r'for symbol in self\._cooldown_tracker:',
        re.MULTILINE
    )
    
    if not check_pattern.search(content):
        # The implementation is missing, we need to add it
        
        # Find where to insert the implementation
        insert_pattern = re.compile(
            r'(    def _update_cooldowns\(self\) -> None:\s*'
            r'"""Decrement cooldown counters\.""")',
            re.MULTILINE
        )
        
        replacement = '''    def _update_cooldowns(self) -> None:
        """Decrement cooldown counters."""
        for symbol in self._cooldown_tracker:
            if self._cooldown_tracker[symbol] > 0:
                self._cooldown_tracker[symbol] -= 1'''
        
        if insert_pattern.search(content):
            content = insert_pattern.sub(replacement, content)
            fixes_applied.append("_update_cooldowns: Added missing implementation")
            print("   ✅ Fixed: Added missing _update_cooldowns implementation")
        else:
            print("   ⚠️  Could not find _update_cooldowns method")
    else:
        print("   ✅ _update_cooldowns implementation already correct")
    
    # =========================================================================
    # FIX 2: Fix on_fill method indentation  
    # =========================================================================
    print("\n🔍 Checking on_fill method indentation...")
    
    # Check if on_fill is at module level (starts at column 0)
    module_level_on_fill = re.compile(r'\ndef on_fill\(self, event: FillEvent\)')
    
    if module_level_on_fill.search(content):
        print("   🔧 Found on_fill at module level - fixing indentation...")
        
        # We need to find the entire on_fill method and reindent it
        lines = content.split('\n')
        new_lines = []
        in_on_fill = False
        method_indent = 0
        brace_depth = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Detect start of module-level on_fill
            if line.startswith('def on_fill(self, event: FillEvent)'):
                in_on_fill = True
                # Add class-level indentation (4 spaces)
                new_lines.append('    ' + line)
                i += 1
                continue
            
            if in_on_fill:
                # Check if we've exited the method
                # A line that's at column 0 and is not blank indicates end of method
                stripped = line.strip()
                
                # Empty lines stay as is
                if not stripped:
                    new_lines.append(line)
                    i += 1
                    continue
                
                # Check if this is a new definition at module level
                if not line.startswith(' ') and not line.startswith('\t'):
                    if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('@'):
                        in_on_fill = False
                        new_lines.append(line)
                        i += 1
                        continue
                
                # Check if we hit another class method (properly indented)
                if line.startswith('    def ') and 'on_fill' not in line:
                    in_on_fill = False
                    new_lines.append(line)
                    i += 1
                    continue
                
                # Add 4 spaces indentation to this line
                new_lines.append('    ' + line)
                i += 1
                continue
            
            new_lines.append(line)
            i += 1
        
        content = '\n'.join(new_lines)
        fixes_applied.append("on_fill: Fixed indentation (moved to class level)")
        print("   ✅ Fixed: on_fill method properly indented")
    else:
        # Check if it's already properly indented
        class_level_on_fill = re.compile(r'\n    def on_fill\(self, event: FillEvent\)')
        if class_level_on_fill.search(content):
            print("   ✅ on_fill method already properly indented")
        else:
            print("   ⚠️  Could not find on_fill method")
    
    # =========================================================================
    # FIX 3: Fix _on_fill method if needed
    # =========================================================================
    print("\n🔍 Checking _on_fill method...")
    
    # Check if _on_fill exists and is properly indented
    if '    def _on_fill(self, event: FillEvent)' in content or '    def _on_fill(self, event):' in content:
        print("   ✅ _on_fill method exists and is properly indented")
    elif 'def _on_fill(' in content:
        print("   ⚠️  _on_fill may have incorrect indentation")
    else:
        # _on_fill might be missing, we should check if it needs to be added
        print("   ℹ️  _on_fill method may need to be added inside on_fill")
    
    # =========================================================================
    # Write the fixed content
    # =========================================================================
    if content != original_content:
        base_path.write_text(content, encoding="utf-8")
        print(f"\n✅ Fixes applied and saved to: {base_path}")
        print("\n📋 Summary of fixes:")
        for fix in fixes_applied:
            print(f"   • {fix}")
    else:
        print("\n✅ No changes needed - file appears to be correct")
    
    # =========================================================================
    # Verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Re-read and verify
    content = base_path.read_text(encoding="utf-8")
    
    issues = []
    
    # Verify _update_cooldowns
    if 'def _update_cooldowns(self) -> None:' in content:
        idx = content.find('def _update_cooldowns(self) -> None:')
        section = content[idx:idx+300]
        if 'for symbol in self._cooldown_tracker:' in section:
            print("✅ _update_cooldowns: VERIFIED - has proper implementation")
        else:
            issues.append("_update_cooldowns still missing implementation")
            print("❌ _update_cooldowns: FAILED - missing implementation")
    
    # Verify on_fill indentation
    if '\ndef on_fill(' in content:
        issues.append("on_fill still at module level")
        print("❌ on_fill: FAILED - still at module level")
    elif '    def on_fill(self, event: FillEvent)' in content:
        print("✅ on_fill: VERIFIED - properly indented as class method")
    
    if issues:
        print("\n⚠️  Some issues remain. Manual intervention may be needed.")
        return 1
    else:
        print("\n🎉 All fixes verified successfully!")
        print("\n📌 Next step: Run 'python diagnose_backtest.py' to test")
        return 0


if __name__ == "__main__":
    sys.exit(main())