def get_center(bbox: list):
    x1, y1, x2, y2 = map(int, bbox)
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2
    return (x_center, y_center)

def get_width(bbox: list) -> int:
    x1, y1, x2, y2 = map(int, bbox)
    return abs(x2 - x1 + 1)

def get_height(bbox: list) -> int:
    x1, y1, x2, y2 = map(int, bbox)
    return abs(y2 - y1 + 1)
