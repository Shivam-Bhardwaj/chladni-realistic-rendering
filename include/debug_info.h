#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DebugInfo {
    unsigned int nan_position_count;
    unsigned int nan_velocity_count;
    unsigned int inf_value_count;
    unsigned int invalid_normal_count;
    unsigned int oob_count;
    unsigned int last_bad_index;
} DebugInfo;

void resetDebugInfo();
void fetchDebugInfo(DebugInfo* hostOut);

#ifdef __cplusplus
}
#endif


