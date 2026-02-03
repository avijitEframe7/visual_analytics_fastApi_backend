CAMERA_CONFIG = {
    "0": {
        "name": "Laptop Camera",
        "type": "laptop",
        "url": None,
        "description": "Built-in laptop webcam"
    },
    "1": {
        "name": "Eframe camera 1",
        "type": "rtsp",
        "url": "rtsp://admin:admin@1966@192.168.100.119:554/cam/realmonitor?channel=4&subtype=0",
        "description": "Eframe camera 1"
    },
    "2": {
        "name": "Eframe camera 2",
        "type": "rtsp",
        "url": "rtsp://admin:admin@1966@192.168.100.119:554/cam/realmonitor?channel=3&subtype=0",
        "description": "Eframe camera 2"
    },
}


def get_rtsp_urls(ids=None):
    """ids=None -> all RTSP (sorted); else URLs for given ids only."""
    if ids is None:
        return [c["url"] for k in sorted(CAMERA_CONFIG) if (c := CAMERA_CONFIG[k]).get("type") == "rtsp" and c.get("url")]
    return [CAMERA_CONFIG[k]["url"] for k in ids if k in CAMERA_CONFIG and (c := CAMERA_CONFIG[k]).get("type") == "rtsp" and c.get("url")]


CAMERA_TYPES = {
    "laptop": "Built-in laptop camera",
    "rtsp": "Network RTSP camera",
    "usb": "USB camera",
    "ip": "IP camera"
}

# Default settings
DEFAULT_CAMERA_ID = "0"
DEFAULT_CAMERA_NAME = "Laptop Camera"
DEFAULT_CAMERA_TYPE = "laptop"
