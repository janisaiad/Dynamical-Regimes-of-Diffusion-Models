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
    output_file.write(f"    \\draw[{color},opacity=0.4] plot coordinates {{\n        ")
    
    # Write coordinates
    coords = []
    for t, x, y in points:
        # Scale coordinates to match desired ranges
        coords.append(f"({t:.2f},{x/2:.4f},{y/2:.4f})")
    
    # Write 8 coordinates per line
    for i in range(0, len(coords), 8):
        output_file.write(" ".join(coords[i:i+8]))
        if i + 8 < len(coords):
            output_file.write("\n        ")
    
    output_file.write("\n    };\n\n")

def write_brownian_tikz(input_file, output_file):
    """Convert Brownian motion data to TikZ plot"""
    try:
        with open(input_file, 'r') as f:
            content = f.read()

        # Split into trajectories
        trajectory_sections = content.split('\n\nTrajectory ')
        
        # Remove header
        trajectory_sections = trajectory_sections[1:]  # Skip everything before first trajectory
        
        # Colors for different trajectories - ensure enough colors
        colors = ['red', 'blue', 'green', 'violet'] * ((len(trajectory_sections) + 3) // 4)
        
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
    % Axe temporel
    \\draw[->] (0,0,0) -- (12,0,0) node[right] {$t$};
    
    % Points temporels importants
    \\draw (8,0,0.1) -- (8,0,-0.1) node[below] {$T=8$};
    \\draw (5,0,0.1) -- (5,0,-0.1) node[below] {$t_s$};
    \\draw (2,0,0.1) -- (2,0,-0.1) node[below] {$t_c$};
    
    % Zones de régimes colorées
    \\fill[blue!20,opacity=0.3] (8,-2,0) -- (8,2,0) -- (5,2,0) -- (5,-2,0) -- cycle; % Régime III
    \\fill[green!20,opacity=0.3] (5,-2,0) -- (5,2,0) -- (2,2,0) -- (2,-2,0) -- cycle; % Régime II
    \\fill[red!20,opacity=0.3] (2,-2,0) -- (2,2,0) -- (0,2,0) -- (0,-2,0) -- cycle; % Régime I
    
    % Labels des régimes
    \\node[above] at (6.5,0,0) {Régime III};
    \\node[above] at (3.5,0,0) {Régime II};
    \\node[above] at (1,0,0) {Régime I};

    % Encarts pour les coupes (y,z) - maintenant perpendiculaires à l'axe du temps
    \\draw[dashed] (1,-1.5,0) -- (1,-1.5,2) -- (1,1.5,2) -- (1,1.5,0) -- cycle; % Encart Régime I
    \\draw[dashed] (3.5,-1.5,0) -- (3.5,-1.5,2) -- (3.5,1.5,2) -- (3.5,1.5,0) -- cycle; % Encart Régime II
    \\draw[dashed] (7,-1.5,0) -- (7,-1.5,2) -- (7,1.5,2) -- (7,1.5,0) -- cycle; % Encart Régime III
""")
            
            # Process each trajectory
            for i, section in enumerate(trajectory_sections):
                points = parse_trajectory(section)
                write_tikz_trajectory(points, colors[i], f)
                
            # Close document
            f.write("\\end{tikzpicture}\n\\end{document}\n")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except IndexError:
        print("Error: Not enough colors for trajectories")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    write_brownian_tikz("beamer_reports/brownian.md", "recap_tikz.tex")
