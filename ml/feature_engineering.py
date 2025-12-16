from rapidfuzz import fuzz
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return 0.0
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def build_features(r1, r2):
    name_sim = fuzz.token_sort_ratio(str(r1.get("name", "")), str(r2.get("name", "")))
    addr_sim = fuzz.token_sort_ratio(str(r1.get("address", "")), str(r2.get("address", "")))
    pin_match = int(r1.get("pincode", "") == r2.get("pincode", ""))
    geo_dist = haversine(
        r1.get("lat"), r1.get("lng"),
        r2.get("lat"), r2.get("lng")
    )
    return [name_sim, addr_sim, pin_match, geo_dist]
