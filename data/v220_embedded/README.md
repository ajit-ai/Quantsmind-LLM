# TAKSH v2.2.0 — Embedded/IoT Domain Data

## Structure

| File | Description |
|---|---|
| `train.jsonl` | Embedded instruction→code pairs (10K–50K) |
| `val.jsonl` | Validation pairs |
| `manifest.json` | Data provenance |

## Categories

- C/C++ microcontroller code
- RTOS task scheduling
- Memory management
- Hardware abstraction layers
- Driver development
- Memory-constrained optimization

## Format

```json
{
  "id": "embedded-001",
  "task": "embedded",
  "instruction": "Write a C function to blink an LED on GPIO pin...",
  "input": "",
  "response": "void blink_led(uint8_t pin, uint32_t delay_ms) {\\n    GPIO_PIN_WRITE(pin, 1);\\n    delay_ms(delay_ms);\\n    GPIO_PIN_WRITE(pin, 0);\\n    delay_ms(delay_ms);\\n}",
  "test_args": [[2, 500]],
  "expected_outputs": [None]
}
```

## Provenance

All training data must be cleared for distribution. See LICENSE_PROVENANCE.md.
