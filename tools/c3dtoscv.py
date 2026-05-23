import c3d
import pandas as pd


def convert_c3d_kinematics_to_csv(c3d_file_path, output_csv_path):
    print(f"Reading C3D file: {c3d_file_path} ...")

    with open(c3d_file_path, "rb") as handle:
        reader = c3d.Reader(handle)

        point_labels = [label.strip() for label in reader.point_labels]

        point_columns = ["Frame"]
        for label in point_labels:
            point_columns.extend([f"{label}_X", f"{label}_Y", f"{label}_Z"])

        point_data_list = []

        for i, points, _ in reader.read_frames():
            frame_point_data = [i]
            for pt in points:
                frame_point_data.extend(pt[:3])
            point_data_list.append(frame_point_data)

        if point_data_list:
            df_points = pd.DataFrame(point_data_list, columns=point_columns)
            df_points.to_csv(output_csv_path, index=False)
            print(f"Successfully saved marker data to: {output_csv_path}")
        else:
            print("No marker data found in the C3D file.")


if __name__ == "__main__":
    INPUT_C3D_FILE = "work-dirs/严九九(YJJ),女,11.23测/c3d/SEG-001/JTDZ01.c3d"
    OUTPUT_CSV_FILE = "trial_01_markers.csv"

    convert_c3d_kinematics_to_csv(INPUT_C3D_FILE, OUTPUT_CSV_FILE)
