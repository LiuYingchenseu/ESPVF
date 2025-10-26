import re
def get_sort_key_patient(filename):
    match = re.match(r'(\d+)_', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # 如果未匹配到数字，将其放在最后

def get_sort_key_frame(filename):
    match = re.match(r'sampled_frame_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # 如果未匹配到数字，将其放在最后
    