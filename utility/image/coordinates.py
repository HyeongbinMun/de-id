def convert_coordinates_to_yolo_format(image_width, image_height, coordinates):
    yolo_labels = []
    for coord in coordinates:
        _, _, _, x, y, w, h = coord
        x, y, w, h = float(x), float(y), float(w), float(h)
        x_center = x + (w / 2)
        y_center = y + (h / 2)
        normalized_x = x_center / image_width
        normalized_y = y_center / image_height
        normalized_w = w / image_width
        normalized_h = h / image_height
        yolo_label = f"0 {normalized_x} {normalized_y} {normalized_w} {normalized_h}"
        yolo_labels.append(yolo_label)
    return yolo_labels