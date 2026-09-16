"""Device resource admission; unsupported geometry is not numerical PASS."""

import ctypes as C


def device_limits(runtime):
    lib = runtime.library()
    get_device = lib.hggcGetDevice
    get_device.argtypes = [C.POINTER(C.c_int)]
    get_attribute = lib.hggcDeviceGetAttribute
    get_attribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
    error = lib.hggcGetErrorString
    error.argtypes, error.restype = [C.c_int], C.c_char_p

    def check(rc):
        if rc:
            raise RuntimeError("device limit query failed: " + error(rc).decode(errors="replace"))

    device = C.c_int()
    check(get_device(C.byref(device)))
    result = {}
    # SDK driver_types.h: hggcDevAttrMaxThreadsPerBlock=1,
    # hggcDevAttrMaxSharedMemoryPerBlockOptin=97 (also checked by host proof).
    for name, attribute in (("max_threads_per_block", 1), ("max_optin_shared_bytes", 97)):
        value = C.c_int()
        check(get_attribute(C.byref(value), attribute, device.value))
        if value.value <= 0:
            raise RuntimeError(f"invalid measured device limit: {name}={value.value}")
        result[name] = value.value
    return result


def resource_reason(stats, limits):
    reasons = []
    if stats["threads"] > limits["max_threads_per_block"]:
        reasons.append("CTA_THREADS_EXCEED_DEVICE_LIMIT")
    if stats["shared_bytes"] > limits["max_optin_shared_bytes"]:
        reasons.append("SHARED_BYTES_EXCEED_DEVICE_OPTIN_LIMIT")
    if reasons:
        return "+".join(reasons)
    if stats["maximum_active_blocks"] < 0:
        # Do not turn an unexplained runtime/registration error into a resource skip.
        raise RuntimeError(f"occupancy query failed despite legal measured limits: {stats}")
    if stats["maximum_active_blocks"] == 0:
        return "OCCUPANCY_API_ZERO_ACTIVE_BLOCKS"
    return None
