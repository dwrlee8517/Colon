#!/usr/bin/env fish
# Usage: ./folder_tree.fish [directory]
# If no directory is provided, it defaults to the current directory.

if test -n "$argv[1]"
    set TARGET_DIR $argv[1]
else
    set TARGET_DIR .
end

find $TARGET_DIR \( -name ".git" -o -name "__pycache__" \) -prune -o -type d -print | sed "s|^$TARGET_DIR/||" | awk -F'/' '
{
  depth = NF - 1;
  indent = "";
  for(i=0; i<depth; i++) {
    indent = indent "    ";
  }
  if($NF != "") {
    print indent "- " $NF;
  }
}'

