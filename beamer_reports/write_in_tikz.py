import re

def parse_trajectory(trajectory_text):
    """Parse a single trajectory from the text into a list of (t,x,y) points"""
    points = []
    for line in trajectory_text.split('\n'):
        if line.startswith('t='):
            # Extract time and coordinates using regex
            match = re.match(r't=(\d+\.\d+): \(([-\d.]+), ([-\d.]+)\)', line)
            if match:
                t = float(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                points.append((t, x, y))
    return points

def write_tikz_trajectory(points, color, output_file):
    """Write a single trajectory as a TikZ path to the output file"""
    # Start path
    output_file.write(f"    \\draw[{color}] plot coordinates {{\n        ")
    
    # Write coordinates
    coords = []
    for t, x, y in points:
        coords.append(f"({t:.2f},{x:.4f},{y:.4f})")
    
    # Write 8 coordinates per line
    for i in range(0, len(coords), 8):
        output_file.write(" ".join(coords[i:i+8]))
        if i + 8 < len(coords):
            output_file.write("\n        ")
    
    output_file.write("\n    };\n\n")

def write_brownian_tikz(input_file, output_file):
    """Convert Brownian motion data to TikZ plot"""
    with open(input_file, 'r') as f:
        content = f.read()

    # Split into trajectories
    trajectory_sections = content.split('\n\nTrajectory ')
    
    # Remove header
    trajectory_sections = trajectory_sections[1:]  # Skip everything before first trajectory
    
    # Colors for different trajectories
    colors = ['red', 'blue', 'green', 'purple']
    
    with open(output_file, 'w') as f:
        # Write preamble
        f.write("""\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\usetikzlibrary{positioning,3d,decorations.markings,calc}
\\pgfplotsset{compat=1.18}

\\begin{document}
\\begin{tikzpicture}[
    x={(0.866cm,-0.5cm)},
    y={(0.866cm,0.5cm)},
    z={(0cm,1cm)}
]
""")
        
        # Process each trajectory
        for i, section in enumerate(trajectory_sections):
            points = parse_trajectory(section)
            write_tikz_trajectory(points, colors[i], f)
            
        # Close document
        f.write("\\end{tikzpicture}\n\\end{document}\n")

if __name__ == "__main__":
    write_brownian_tikz("brownian.md", "recap_tikz.tex")
