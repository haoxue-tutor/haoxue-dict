# haoxue-dict

A Chinese dictionary and word segmenter.

## Dictionary usage

```rust
use haoxue_dict::DICTIONARY;

let entry = DICTIONARY.get_entry("你好").unwrap();
assert_eq!(entry.simplified(), "你好");
assert_eq!(entry.pinyin(), "ni3 hao3");
assert_eq!(prettify_pinyin::prettify(entry.pinyin()), "nǐ hǎo");
```

```rust
use haoxue_dict::DICTIONARY;

let entry = DICTIONARY.get_entry("们").unwrap();
assert_eq!(entry.traditional(), "們");
assert_eq!(entry.pinyin(), "men5");
assert_eq!(prettify_pinyin::prettify(entry.pinyin()), "men");
```

```rust
use haoxue_dict::DICTIONARY;

// 们 is more common than 大学
assert!(DICTIONARY.frequency("们") > DICTIONARY.frequency("大学"));
```

## Segmenter usage

```rust
use haoxue_dict::{DICTIONARY, DictEntry};
use either::Either;

let segments = DICTIONARY.segment("明天我会去图书馆。")
                .iter()
                .map(|segment| segment.map_left(DictEntry::simplified))
                .collect::<Vec<_>>();
assert_eq!(segments, vec![
    Either::Left("明天"),
    Either::Left("我"),
    Either::Left("会"),
    Either::Left("去"),
    Either::Left("图书馆"),
    Either::Right("。")
]);
```

## Feature flags

- `embed-dict`: Embed the dictionary in the binary. This is the default feature and adds about 12.4 MiB to the binary size.
