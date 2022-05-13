"""
Pack spheres in a cylindrical container.
"""

import numpy as np
from numpy import pi
import bisect
import os
import gzip
import time

SMALL = 1e-6


class Sph:
    def __init__(self, r):
        self.r = r

    def __lt__(self, b):
        # Sort using the z coordinate
        return self.r[2] < b.r[2]


def norm(x):
    return np.sqrt(norm2(x))


def norm2(x):
    return np.inner(x, x)


def debug(x):
    print(x)


def verbose(x):
    print(x)


class Spheres:

    def __init__(self, sph_rad, cyl_rad, cyl_height, dx=1., ior=1., fill_now=False):
        """
        Spheres in a cylinder.

        sph_rad = sphere radius
        cyl_rad = cylinder (container) radius
        cyl_height = cylinder (container) height
        dx = voxel dimensions for slice generations
        ior = index of refraction difference with surrounding (default 1.)

        Notes:
            - Sphere all have the same radius
            - Slow implementation
            - Gives only sagittal slices and projections.

        Example:
        A 50 micron diameter container with 1 micron spheres. (.1 micron pixel size)

        S = Spheres(sph_rad=.5e-6,
                    cyl_rad=25e-6,
                    cyl_height=100e-6,
                    dx=.1e-6,
                    fill_now=True)
        # Or with fill_now=False, one needs then to call `S.stack_fill()`.

        S.get_slice(0.) -> central (sagittal) slice
        S.get_slice(-10e-6) -> sagittal slice 10 micron off the center
        S.get_proj() -> integrated density
        S.to_angle(angle) -> rotate the spheres

        To generate a tomographic dataset (pseudo-code):
        for angle in some_angle_list:
            S.to_angle(angle)
            proj.append(S.get_proj())
        """
        self.sph_rad = sph_rad
        self.cyl_height = cyl_height
        self.cyl_rad = cyl_rad
        self.dx = dx
        self.angle = 0.
        self.ior = ior
        self.pos = None
        if fill_now:
            self.stack_fill()

    def stack_fill(self):
        """\
        Fill the container by stacking spheres from a random bottom layer.
        """

        R = self.sph_rad        # sphere radius
        D = 2*R                 # sphere diameter
        D2 = D*D
        cylR = self.cyl_rad
        cylR2 = cylR**2

        # height of first container (for initial random distribution)
        cylH = 3*R

        thresh_dist = SMALL*R
        D2p = D2 + thresh_dist
        D2m = D2 - thresh_dist

        # First generate bottom layer randomly
        # debug('Generating initial layer...')
        new_container = Spheres(R, cylR, cylH, self.dx, fill_now=False)
        new_container.random_fill(Ntry_limit=5000)
        pos = new_container.pos
        del new_container
        pos.sort()

        zmin = 0
        zmax = self.cyl_height

        # Dictionary of nearest neighbors
        neighbor = {}

        # List of possible next position, ordered in z
        next_pos = []

        # Prepare initial table
        verbose('Finding initial site list...')
        for r1 in pos:
            nn = []
            for r2 in pos:
                if r2 is r1:
                    continue
                if norm2(r2.r - r1.r) < 4*D2p:
                    nn.append(r2)

                for r3 in pos:
                    if (r3 is r1) or (r3 is r2):
                        continue
                    r = fourth_kiss(r1.r, r2.r, r3.r, R)
                    if r:
                        new_pt = Sph(r[0])

                        # Skip if this is a duplicate
                        if any([norm2(new_pt.r - x.r) < thresh_dist for x in next_pos]):
                            continue

                        # Add if no overlap with other positions
                        if all([(norm2(x.r-r[0]) > D2m) for x in pos + next_pos]):
                            bisect.insort(next_pos, new_pt)

                        if all([(norm2(x.r-r[1]) > D2m) for x in pos + next_pos]):
                            bisect.insort(next_pos, Sph(r[1]))

            neighbor[r1] = nn
        verbose('Found {0} initial sites'.format(len(next_pos)))

        verbose('Starting main loop.')
        time_info = []

        while True:

            if len(pos) % 10 == 0:
                time_info.append(time.time())
                verbose(
                    '{0} spheres / {1} sites [{2}]'.format(len(pos), len(next_pos), time.ctime()))

            if not next_pos:
                break

            while True:
                # Pick lowest next point
                new_sphere = next_pos.pop(0)
                # exit loop if sphere is inside the cylinder
                if (new_sphere.r[2] > zmin) and (norm2(new_sphere.r[:2]) < cylR2):
                    break

            # Max height reached - we are done
            if new_sphere.r[2] > zmax:
                break

            # Remove the possible positions that overlap with this new choice
            len_before = len(next_pos)
            next_pos = [xn for xn in next_pos if (
                norm2(new_sphere.r - xn.r) > D2m)]
            len_after = len(next_pos)
            removed = len_before - len_after
            verbose(f"{removed} overlapping sites removed (out of {len_before})")

            # Update neighbor list
            nn = []
            nr = new_sphere.r
            # Reversed loop - there is no point going below D
            for r1 in reversed(pos):
                if (r1.r[2] - nr[2]) < -2*D:
                    break
                if norm2(r1.r - nr) < 4*D2:
                    nn.append(r1)
                    neighbor[r1].append(new_sphere)
            neighbor[new_sphere] = nn

            verbose(
                f"New sphere at ({new_sphere.r[0]:.02f}, {new_sphere.r[1]:.02f}, {new_sphere.r[2]:.02f})")
            # Keep sphere positions sorted vertically
            bisect.insort(pos, new_sphere)

            # Find new sites
            for r1 in neighbor[new_sphere]:
                for r2 in neighbor[new_sphere]:
                    if r1 is r2:
                        continue
                    r = fourth_kiss(r1.r, r2.r, new_sphere.r, R)
                    if r:
                        if all([(norm2(x.r - r[0]) > D2m) for x in set(neighbor[r1] + neighbor[r2] + neighbor[new_sphere])]):
                            bisect.insort(next_pos, Sph(r[0]))

                        if all([(norm2(x.r - r[1]) > D2m) for x in set(neighbor[r1] + neighbor[r2] + neighbor[new_sphere])]):
                            bisect.insort(next_pos, Sph(r[1]))

            added = len(next_pos) - len_after
            verbose(f"Found {added} new sites.")

            # Add the new sphere to the stack
            # pos.append(new_sphere)

        # Refilter to make sure all spheres are really within the container
        self.pos = [x for x in pos if ((x.r[2] >= R)
                    and (x.r[2] <= self.cyl_height-R)
                    and (norm(x.r[:2]) < (self.cyl_rad - R)))]
        verbose('There were {0} spheres in the list, after removing those that spilled over, there are {1}.'.format(
            len(pos), len(self.pos)))
        self.time_info = time_info
        cyl_vol = pi * self.cyl_height * self.cyl_rad**2
        sph_vol = 4*pi*self.sph_rad**3 / 3
        N = len(self.pos)
        density = N*sph_vol/cyl_vol
        verbose("Stacking complete.")
        verbose(
            f"There are {N} spheres in the container (density = {density:.04f})")

    def random_fill(self, Ntry_limit=None):
        """\
        Fill the container with spheres, using a Monte Carlo like approach.
        """
        # Container and sphere volumes
        cyl_vol = pi * self.cyl_height * self.cyl_rad**2
        sph_vol = 4*pi*self.sph_rad**3 / 3

        # This is roughly the maximum number of spheres that can fit in
        Nmax = (pi / np.sqrt(18.)) * cyl_vol / sph_vol

        verbose("Rough upper bound of number of sphere: {Nmax}")

        if Ntry_limit is None:
            Ntry_limit = Nmax

        # Container dimensions
        box = np.asarray([2*(self.cyl_rad-self.sph_rad), 2
                         * (self.cyl_rad-self.sph_rad), self.cyl_height-2*self.sph_rad])

        pos = []
        Ntry = 0
        d2 = 4 * self.sph_rad**2  # squared diameter
        ctr_offset = .5*box
        ctr_offset[2] = 0.
        while True:
            # Stopping criterion
            if Ntry > Ntry_limit:
                break

            # New sphere coordinate
            x = np.random.rand(3)*box - ctr_offset

            # Check if in cylinder
            if (norm2(x[:2]) > (self.cyl_rad - self.sph_rad)**2):
                continue

            # Check if overlap with other spheres
            failed = False
            for xs in pos:
                if norm2(xs.r-x) < d2:
                    failed = True
                    break
            if failed:
                Ntry += 1
            else:
                pos.append(Sph(x))
                verbose(
                    f"New sphere at ({x[0]:.02f}, {x[1]:.02f}, {x[2]:.02f})! (tried {Ntry} times)")
                Ntry = 0

        self.angle = 0.
        self.pos = pos
        N = len(pos)
        density = N*sph_vol/cyl_vol
        verbose("Filling complete.")
        verbose(
            f"There are {N} spheres in the container (density = {density:.04f})")

    def save_coords(self, filename):
        """\
        Save the spheres coordinates (rescaled such that the radius is 1) in a text file.
        """
        filename = os.path.abspath(os.path.expanduser(filename)) + '.gz'
        with gzip.open(filename, 'wb') as f:
            for x in self.pos:
                f.write(b'%.12e   %.12e   %.12e\n' % tuple(x.r / self.sph_rad))
            verbose(f'Sphere coordinates written to {filename}')

    def load_coords(self, filename):
        """\
        Reads the sphere coordinates from file.
        """
        # Take care of gzipped files, etc.
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filename):
            filename += '.gz'
            if not os.path.exists(filename):
                raise IOError('file does not exists.')
        pts = np.loadtxt(filename)

        # Rescale coordinates to sphere size
        pts *= self.sph_rad

        # Check if container size is OK
        max_coords = pts.max(axis=0)
        cyl_rad = self.sph_rad + np.ceil(max_coords[0])
        cyl_height = self.sph_rad + np.ceil(max_coords[2])
        if cyl_rad != self.cyl_rad:
            verbose(
                f'Loaded coordinates indicate that the cylinder radius should be {cyl_rad}')
            verbose(f'instead of {self.cyl_rad}')
            verbose(f'Changing the internal value to {cyl_rad}.')
            self.cyl_rad = cyl_rad
        if cyl_height != self.cyl_height:
            verbose(
                f'Loaded coordinates indicate that the cylinder height should be {cyl_height}')
            verbose(f'instead of {self.cyl_height}')
            verbose(f'Changing the internal value to {cyl_height}')
            self.cyl_height = cyl_height
        self.pos = [Sph(x) for x in pts]

    def rotate_by(self, angle):
        """\
        Rotate the sample by angle (in degrees).
        """
        self.angle += angle
        sinth = np.sin(pi*angle/180)
        costh = np.cos(pi*angle/180)
        transf = np.matrix(np.eye(3))
        transf[:2, :2] = [[costh, sinth], [-sinth, costh]]
        for x in self.pos:
            x.r = (x.r * transf).A1

    def rotate_to(self, angle):
        """\
        Rotate the sample to given angle (in degrees).
        """
        self.rotate_by(angle - self.angle)

    def get_coords(self):
        return np.array([x.r for x in self.pos])

    def get_slice(self, dist, dx=None, out=None):
        """\
        Generates a slice through the container at a distance dist from the origin,
        along the current y axis.
        """
        if self.pos is None:
            return None

        if dx is not None:
            self.dx = dx

        # create output array
        sh = (int(2*self.cyl_rad/self.dx), int(self.cyl_height/self.dx))
        if out is None:
            out = np.zeros(sh)

        # Offset of the local array for a single sphere
        l_offset = int(np.ceil(self.sph_rad/self.dx))

        # Size of local array
        sh1 = (2*l_offset + 1, 2*l_offset + 1)

        # Coordinates
        xx, zz = np.indices(sh1)

        # Slice position (voxel units)
        y_slice = dist/self.dx
        # Sphere radius (voxel units)
        R = self.sph_rad/self.dx
        R2 = R**2
        for xs in self.pos:
            x, y, z = xs.r / self.dx
            s = abs(y-y_slice)
            x += self.cyl_rad / self.dx
            if s >= R:
                # sphere is out of slice
                continue

            # radius of the sphere cut
            rslice2 = R2 - s**2

            # local array corner position
            lpos_x, lpos_z = int(
                np.floor(x)-l_offset), int(np.floor(z)-l_offset)
            # sphere center in the local array
            lctr_x, lctr_z = x-lpos_x, z-lpos_z
            out[lpos_x:(lpos_x+sh1[0]), lpos_z:(lpos_z+sh1[1])
                ] += ((xx - lctr_x)**2 + (zz - lctr_z)**2) < rslice2

        return self.ior * out

    def get_proj(self, dx=None, out=None):
        """\
        Generates the projection through the container.
        """
        if self.pos is None:
            return None

        if dx is not None:
            self.dx = dx
        # create output array
        sh = (int(2*self.cyl_rad/self.dx), int(self.cyl_height/self.dx))
        if out is None:
            out = np.zeros(sh)

        # Offset of the local array for a single sphere
        l_offset = int(np.ceil(self.sph_rad/self.dx))

        # Size of local array
        sh1 = (2*l_offset + 1, 2*l_offset + 1)

        # Coordinates
        xx, zz = np.indices(sh1)

        # Sphere radius (voxel units)
        R = self.sph_rad/self.dx
        R2 = R**2

        for xs in self.pos:
            x, y, z = xs.r / self.dx
            x += self.cyl_rad / self.dx
            # local array corner position
            lpos_x, lpos_z = int(
                np.floor(x)-l_offset), int(np.floor(z)-l_offset)
            # sphere center in the local array
            lctr_x, lctr_z = x-lpos_x, z-lpos_z
            r2pos = R2 - (xx - lctr_x)**2 - (zz - lctr_z)**2
            if out[lpos_x:(lpos_x+sh1[0]), lpos_z:(lpos_z+sh1[1])].shape != sh1:
                1/0
            out[lpos_x:(lpos_x+sh1[0]), lpos_z:(lpos_z+sh1[1])
                ] += 2*np.sqrt(r2pos * (r2pos > 0))

        return self.ior * out

    def __getitem__(self, i):
        """\
        This returns getslice(self.dx*i - self.cyl_rad), or IndexError if
        i < 0 or i > 2*self.cyl_rad / self.dx
        """
        dist = self.dx*i - self.cyl_rad
        if abs(dist) > self.cyl_rad:
            raise IndexError()
        return self.get_slice(dist)

    def __len__(self):
        return int(2*self.cyl_rad / self.dx) + 1

    def __repr__(self):
        out = "<spheres.Spheres object>\n" +\
              "sphere radius = %.3f\n" +\
              "cylinder radius/height = %.3f/%.3f\n"
        out = out % (self.sph_rad, self.cyl_rad, self.cyl_height)
        if self.pos is None:
            out += "total number of spheres = 0"
        else:
            out += "total number of spheres = %d" % len(self.pos)
        return out


def fourth_kiss(r1, r2, r3, radius):
    """\
    returns the two position vectors of a sphere of given radius which kisses at once
    all the spheres at positions r1,r2 and r3. None is returned if this is impossible.
    """

    D = 2*radius

    s2 = r2 - r1
    s3 = r3 - r1

    ds2 = norm(s2)
    ds3 = norm(s3)

    if (ds2 > 2*D) or (ds3 > 2*D) or (norm2(r3 - r2) > 4*D*D):
        return None

    c = sum(s2*s3)/(ds2*ds3)

    d = .5 / (1-c**2) * (s2 * (1-c*ds3/ds2) + s3 * (1-c*ds2/ds3))

    d2 = norm2(d)
    if D*D < d2:
        return None

    dl = np.sqrt(D*D - d2)

    uperp = np.cross(s2, s3)
    uperp /= norm(uperp)

    r4a = r1 + d + uperp*dl
    r4b = r1 + d - uperp*dl

    return (r4a, r4b)
