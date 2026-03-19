from pathlib import Path

def extract_manuscript_id(filename):
    """
    Extract manuscript ID from your filename format
    
    Examples:
        "CH-Fco Ms. 2_006r copy.jpg" -> "CH-Fco Ms. 2"
        "D-KNd 1161 032r.jpg" -> "D-KNd 1161"
        "NZ-Wt MSR-03 013r.png" -> "NZ-Wt MSR-03"
    """
    stem = Path(filename).stem
    
    # Replace underscores with spaces for consistent splitting
    stem = stem.replace('_', ' ')
    parts = stem.split()
    
    # Find where page number starts (digits + r/v)
    manuscript_parts = []
    for part in parts:
        # Stop at page numbers like "006r", "112v", "032r"
        if any(c.isdigit() for c in part) and len(part) >= 3:
            if part[-1] in ['r', 'v'] or part[-2:] == 'copy':
                break
        manuscript_parts.append(part)
    
    if manuscript_parts:
        return ' '.join(manuscript_parts)
    
    # Fallback: first 2 parts
    return ' '.join(parts[:2]) if len(parts) >= 2 else stem

# Test it
if __name__ == '__main__':
    test_files = [
        "CH-Fco Ms. 2_006r copy.jpg",
        "D-KNd 1161 032r.jpg",
        "NZ-Wt MSR-03 013r.png",
        "CH-P 18 p.100.jpg"
    ]
    
    for f in test_files:
        print(f"{f:40s} -> {extract_manuscript_id(f)}")
