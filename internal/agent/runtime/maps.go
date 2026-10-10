package runtime

import "strconv"

// FirstStringFromMap returns the first key's value as a string, trying each
// key in order; "" when none match or are non-string.
func FirstStringFromMap(m map[string]any, keys ...string) string {
	if m == nil {
		return ""
	}
	for _, key := range keys {
		if v, ok := m[key]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	return ""
}

// IntFromMap returns the int value for key, or -1 when absent/invalid.
func IntFromMap(m map[string]any, key string) int {
	if m == nil {
		return -1
	}
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case int:
			return n
		case int64:
			return int(n)
		case float64:
			return int(n)
		case string:
			if i, err := strconv.Atoi(n); err == nil {
				return i
			}
		}
	}
	return -1
}
