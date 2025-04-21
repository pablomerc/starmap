import starry
import numpy as np
import os
import time as timer
import random
import argparse
from tqdm import tqdm

starry.config.lazy = False
starry.config.quiet = True

# --- Parameter Ranges ---
YDEG, UDEG = 30, 2
U1_RANGE, U2_RANGE = [0.1, 0.3], [0.1, 0.2]
NUM_SPOTS_RANGE = [1, 3]
SPOT_CONTRAST_RANGE = [0.2, 0.8]
SPOT_RADIUS_RANGE = [3.0, 15.0]
SPOT_LAT_RANGE, SPOT_LON_RANGE = [0, 0], [-90.0, 90.0]
RO_RANGE = [0.05, 0.3]
XO_START_RANGE, XO_END_RANGE = [-2.0, -1.5], [1.5, 2.0]
YO_CONST_RANGE, ZO_CONST = [0, 0], 1.0
N_PTS, TIME_START, TIME_END = 1000, 0.0, 1.0
IMG_RESOLUTION, SAVE_IMAGES = 256, True

parameter_ranges = {
    'u1': U1_RANGE, 'u2': U2_RANGE,
    'num_spots': NUM_SPOTS_RANGE,
    'spot_contrast': SPOT_CONTRAST_RANGE,
    'spot_radius': SPOT_RADIUS_RANGE,
    'spot_lat': SPOT_LAT_RANGE,
    'spot_lon': SPOT_LON_RANGE,
    'ro': RO_RANGE,
    'xo_start': XO_START_RANGE,
    'xo_end': XO_END_RANGE,
    'yo_const': YO_CONST_RANGE
}

def generate_synthetic_example(param_ranges):
    params = {}
    map_obj = starry.Map(ydeg=YDEG, udeg=UDEG)

    u1, u2 = random.uniform(*param_ranges['u1']), random.uniform(*param_ranges['u2'])
    map_obj[1], map_obj[2] = u1, u2
    params['u1'], params['u2'] = u1, u2

    num_spots = random.randint(*param_ranges['num_spots'])
    params['num_spots'] = num_spots
    spots = []
    for _ in range(num_spots):
        contrast = random.uniform(*param_ranges['spot_contrast'])
        radius = random.uniform(*param_ranges['spot_radius'])
        lat = random.uniform(*param_ranges['spot_lat'])
        lon = random.uniform(*param_ranges['spot_lon'])
        map_obj.spot(contrast=contrast, radius=radius, lat=lat, lon=lon)
        spots.append({'contrast': contrast, 'radius': radius, 'lat': lat, 'lon': lon})
    params['spots'] = spots

    y_coeffs = map_obj.y
    img_array = None
    if SAVE_IMAGES:
        try:
            img_array = map_obj.render(res=IMG_RESOLUTION, projection='ortho', theta=0.0)
        except Exception:
            img_array = np.full((IMG_RESOLUTION, IMG_RESOLUTION), np.nan)

    ro = random.uniform(*param_ranges['ro'])
    xo_start, xo_end = random.uniform(*param_ranges['xo_start']), random.uniform(*param_ranges['xo_end'])
    yo_const = random.uniform(*param_ranges['yo_const'])

    params['ro'] = ro
    params['trajectory'] = {
        'xo_start': xo_start, 'xo_end': xo_end,
        'yo_const': yo_const, 'zo_const': ZO_CONST
    }

    time_lc = np.linspace(TIME_START, TIME_END, N_PTS)
    xo = np.linspace(xo_start, xo_end, N_PTS)
    yo = np.full(N_PTS, yo_const)
    zo = np.full(N_PTS, ZO_CONST)

    flux = map_obj.flux(xo=xo, yo=yo, ro=ro, zo=zo)
    return time_lc, flux, img_array, y_coeffs, params

def main(n_examples, examples_per_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {n_examples} examples. Saving every {examples_per_file} to {output_dir}/")

    all_times, all_fluxes, all_images, all_y_coeffs, all_params = [], [], [], [], []
    file_counter = 0
    start = timer.time()

    for i in tqdm(range(n_examples), desc="Generating examples"):
        time_lc, flux, img, y_coeffs, params = generate_synthetic_example(parameter_ranges)
        all_times.append(time_lc)
        all_fluxes.append(flux)
        if SAVE_IMAGES:
            all_images.append(img)
        all_y_coeffs.append(y_coeffs)
        all_params.append(params)

        if (i + 1) % examples_per_file == 0 or (i + 1) == n_examples:
            chunk_fname = os.path.join(output_dir, f"synthetic_starry_data_part_{file_counter}.npz")
            save_dict = {
                'time': np.array(all_times, dtype=np.float32),
                'flux': np.array(all_fluxes, dtype=np.float32),
                'y_coeffs': np.array(all_y_coeffs, dtype=np.float32),
                'parameters': np.array(all_params, dtype=object)
            }
            if SAVE_IMAGES:
                if all(isinstance(im, np.ndarray) for im in all_images):
                    save_dict['image'] = np.array(all_images, dtype=np.float32)
                else:
                    save_dict['image'] = np.array(all_images, dtype=object)
            np.savez_compressed(chunk_fname, **save_dict)
            file_counter += 1
            all_times, all_fluxes, all_images, all_y_coeffs, all_params = [], [], [], [], []

    elapsed = timer.time() - start
    print(f"✅ Done! Generated {n_examples} examples in {elapsed:.2f} seconds.")
    print(f"Saved in {file_counter} file(s) in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic starry light curves.")
    parser.add_argument("n_examples", type=int, help="Total number of examples to generate.")
    parser.add_argument("examples_per_file", type=int, help="Number of examples per output file.")
    parser.add_argument("output_dir", type=str, help="Directory to save output .npz files.")
    args = parser.parse_args()

    main(args.n_examples, args.examples_per_file, args.output_dir)

