CAMERA_CONFIG = {
    "1": {
        "name": "Eframe Camera 1",
        "type": "rtsp",
        "url": "rtsp://admin:admin@1966@192.168.100.119:554/cam/realmonitor?channel=4&subtype=0",
        "description": "Eframe Camera 1",
    },
    "2": {
        "name": "Eframe Camera 2",
        "type": "rtsp",
        "url": "rtsp://admin:admin@1966@192.168.100.119:554/cam/realmonitor?channel=3&subtype=0",
        "description": "Eframe Camera 2",
    },
}


def get_rtsp_urls(ids=None):
    """ids=None -> all RTSP (sorted); else URLs for given ids only. Keys normalized to string for lookup."""
    if ids is None:
        return [c["url"] for k in sorted(CAMERA_CONFIG) if (c := CAMERA_CONFIG[k]).get("type") == "rtsp" and c.get("url")]
    return [CAMERA_CONFIG[str(k)]["url"] for k in ids if str(k) in CAMERA_CONFIG and (c := CAMERA_CONFIG[str(k)]).get("type") == "rtsp" and c.get("url")]


CAMERA_TYPES = {
    "laptop": "Built-in laptop camera",
    "rtsp": "Network RTSP camera",
    "usb": "USB camera",
    "ip": "IP camera",
}
