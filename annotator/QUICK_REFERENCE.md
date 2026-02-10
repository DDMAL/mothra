# Quick Reference Card

Print this or keep it handy while annotating!

## Starting Up

```bash
python annotate_yolo.py --image your_manuscript.png
```

## Controls Cheat Sheet

```
┌─────────────────────────────────────────────┐
│  CLASS SELECTION                            │
├─────────────────────────────────────────────┤
│  1  →  text                                 │
│  2  →  music                                │
│  3  →  staves                               │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ANNOTATION                                 │
├─────────────────────────────────────────────┤
│  Click & Drag  →  Draw bounding box         │
│  u / Ctrl+Z    →  Undo last annotation      │
│  d             →  Delete (hover first)      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  SAVING                                     │
├─────────────────────────────────────────────┤
│  e  →  Export to YOLO format                │
│  q  →  Quit and save                        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  OTHER                                      │
├─────────────────────────────────────────────┤
│  ESC  →  Cancel current box                 │
│  s    →  Force save (usually automatic)     │
└─────────────────────────────────────────────┘
```

## Annotation Workflow

1. Press **number key** to select class
2. **Click and drag** to draw box
3. Box **saves automatically**
4. Repeat for all objects
5. Press **e** to export
6. Press **q** to quit

## Tips

✓ Be consistent with box sizes
✓ Include the entire object in the box
✓ Don't worry about perfect boxes - good enough is fine
✓ Use undo (u) liberally if you make mistakes
✓ Export (e) periodically to save your work
✓ Sessions auto-save - you can quit anytime

## Color Guide

- **Blue boxes** = text
- **Green boxes** = music
- **Red boxes** = staves

Thick outline = hovering (ready to delete)

## Common Questions

**Q: Do I need to annotate every single instance?**
A: Yes, for best results annotate all visible instances of each class.

**Q: Can I overlap boxes?**
A: Yes, YOLO handles overlapping boxes fine.

**Q: What if I quit accidentally?**
A: No problem! Your session auto-saves. Just run the same command again.

**Q: How do I know it worked?**
A: Check `annotations/yolo_format/` - you should see a `.txt` file with numbers.

**Q: Where do the annotations go?**
A: `annotations/` folder in the same directory as the script.