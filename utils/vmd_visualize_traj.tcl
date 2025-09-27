set pdbfile   [lindex $argv 0]
set giffile   [lindex $argv 1]
set tmpdir   [lindex $argv 2]
set speed   [lindex $argv 3]
set duration   [lindex $argv 4]

# Create a temporary file to store rendered frames.
file mkdir $tmpdir

# Set play speed
animate speed $speed
# Set the movie duration
set movie_duration $duration

# Load molecule ----------------------------------------
mol new $pdbfile type pdb waitfor all

# Remove default representation
mol delrep 0 top

# Modify representation ---------------------------------
# Add NewRibbons representation, color by chain
mol representation NewCartoon
mol color Chain
mol addrep top

# Modify display ----------------------------------------
# Background white, orthographic projection
color Display Background white
display projection orthographic

# Align trajectory ---------------------------------------
# Align the trajectory to the first frame
set ref [atomselect top "backbone" frame 0]
set sel [atomselect top "backbone"]

set n [molinfo top get numframes]
for {set i 0} {$i < $n} {incr i} {
    $sel frame $i
    $sel move [measure fit $sel $ref]
}

$ref delete
$sel delete

# Make a movie ----------------------------------------
# Render frames from a trajectory
for {set i 0} {$i < $n} {incr i} {
    animate goto $i
    # Path to the rendered frame.
    set imgfile [format "%s/frame%04d.tga" $tmpdir $i]
    render TachyonInternal $imgfile
}

# Create movie
set fps [expr {$n / $movie_duration}]
# exec convert -delay [expr {100/$fps}] -loop 0 $tmpdir/frame*.tga $giffile
exec magick convert -delay [expr {100/$fps}] -loop 0 $tmpdir/frame*.tga $giffile

# Clean up
file delete -force $tmpdir
exit

