# Trimmed speaker feature report

规则：提取第一个完整 `Response`，删除后续续写；异常样本输出为空字符串。

| File | Records | Trimmed | No response | Too short | Too long | First response polluted |
|---|---:|---:|---:|---:|---:|---:|
| `iemocap.test_spdescV6_Qwen2.5-7B-Instruct.json` | 1623 | 1532 | 91 | 0 | 0 | 0 |
| `iemocap.train_spdescV6_Qwen2.5-7B-Instruct.json` | 5163 | 5163 | 0 | 0 | 0 | 0 |
| `iemocap.valid_spdescV6_Qwen2.5-7B-Instruct.json` | 647 | 647 | 0 | 0 | 0 | 0 |
| `meld.test_spdescV6_Qwen2.5-7B-Instruct.json` | 2610 | 2610 | 0 | 0 | 0 | 0 |
| `meld.train_spdescV6_Qwen2.5-7B-Instruct.json` | 9989 | 9988 | 0 | 1 | 0 | 0 |
| `meld.valid_spdescV6_Qwen2.5-7B-Instruct.json` | 1109 | 1109 | 0 | 0 | 0 | 0 |

## Examples

### `iemocap.test_spdescV6_Qwen2.5-7B-Instruct.json`

- `no_response`: `[{"conversation": "Ses05F_impro05", "index": 0, "response_count": 0, "reason": null, "raw_chars": 12019}, {"conversation": "Ses05F_impro05", "index": 1, "response_count": 0, "reason": null, "raw_chars": 12399}, {"conversation": "Ses05F_impro05", "index": 2, "response_count": 0, "reason": null, "raw_chars": 12019}, {"conversation": "Ses05F_impro05", "index": 3, "response_count": 0, "reason": null, "raw_chars": 12399}, {"conversation": "Ses05F_impro05", "index": 4, "response_count": 0, "reason": null, "raw_chars": 12019}]`

### `iemocap.train_spdescV6_Qwen2.5-7B-Instruct.json`


### `iemocap.valid_spdescV6_Qwen2.5-7B-Instruct.json`


### `meld.test_spdescV6_Qwen2.5-7B-Instruct.json`


### `meld.train_spdescV6_Qwen2.5-7B-Instruct.json`

- `too_short`: `[{"conversation": "162", "index": 1, "response_count": 6, "reason": "only_1_words", "raw_chars": 3895}]`

### `meld.valid_spdescV6_Qwen2.5-7B-Instruct.json`


