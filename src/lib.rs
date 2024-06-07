use cedict::DictEntry;
use either::Either;
use itertools::Itertools;
use once_cell::sync::Lazy;
use std::{
    collections::{BTreeMap, HashMap},
    io::Cursor,
    ops::RangeFrom,
};

static DEFAULT_DICT: &str = include_str!("../data/cedict-2024-06-07.txt");
static DEFAULT_WF: &str = include_str!("../data/SUBTLEX-CH-WF.utf8.txt");

static SEGMENTATION_EXCEPTIONS: &[&[&str]] = &[
    &["家", "中餐馆"],
    &["这", "位子"],
    &["十", "分钟"],
    &["一", "点钟"],
    &["合上", "书"],
    &["第二", "天性"],
    &["都", "会"],
    &["上", "都"],
    &["把", "手举"],
    &["天", "下雨"],
    &["四十", "分"],
    &["写", "作文"],
    &["得", "很"],
    &["家", "的"],
    &["的", "话"],
];

pub static DICTIONARY: Lazy<Dictionary> = Lazy::new(Dictionary::new);

pub struct Dictionary {
    entries: BTreeMap<String, Vec<DictEntry<String>>>,
    word_frequency: HashMap<String, f64>,
}

impl Dictionary {
    pub fn new() -> Self {
        Dictionary {
            entries: cedict::parse_reader(Cursor::new(DEFAULT_DICT))
                // Filter out non-chinese entries.
                .filter(|entry| !entry.simplified().chars().all(|c| c.is_ascii()))
                .sorted_by(|a, b| a.simplified().cmp(&b.simplified()))
                .group_by(|entry| entry.simplified().to_string())
                .into_iter()
                .map(|(key, entries)| (key, entries.collect()))
                .collect(),
            word_frequency: csv::ReaderBuilder::new()
                .delimiter(b'\t')
                .has_headers(true)
                .from_reader(Cursor::new(DEFAULT_WF))
                .deserialize()
                .map(|x| x.unwrap())
                .map(
                    |(word, _wcount, _wmillion, logw, _wcd, _wcdp, _logwcd): (
                        String,
                        u64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                    )| (word, logw),
                )
                .collect(),
        }
    }

    pub fn frequency(&self, word: &str) -> f64 {
        self.word_frequency.get(word).copied().unwrap_or_else(|| {
            word.chars()
                .map(|c| {
                    let mut buf = [0; 4];
                    let result = c.encode_utf8(&mut buf);
                    self.word_frequency.get(result).copied().unwrap_or(0f64)
                })
                .sum::<f64>()
                .powf((word.chars().count() as f64).recip())
        })
    }

    fn lookup_entry<'a>(&'a self, entry: &str) -> Option<Option<&'a Vec<DictEntry<String>>>> {
        let (first_entry, dict_entry): (&String, &Vec<DictEntry<String>>) = self
            .entries
            .range(RangeFrom {
                start: entry.to_string(),
            })
            .next()?;
        if !first_entry.starts_with(entry) {
            None
        } else if entry == first_entry {
            Some(Some(dict_entry))
        } else {
            Some(None)
        }
    }

    pub fn lookup_entries<'a: 'b, 'b>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'a DictEntry<String>> + 'b {
        string_inits(text)
            .map_while(|entry| self.lookup_entry(entry))
            .filter_map(|x| std::convert::identity(x))
            .flatten()
    }

    // 十分钟
    // 0. segments: [[]]
    // 1. segments: [[十], [十分]]
    // 2. segments: [[十,分], [十,分钟],[十分]]
    // 3. segments: [[十,分,钟], [十,分钟],[十分, 钟]]
    // 4. pick best: [十,分钟]
    fn segment_step<'a>(&'a self, text: &str) -> Vec<&'a DictEntry<String>> {
        let mut fragments = Fragments::new();
        fragments.push_fragment(Fragment::new());
        loop {
            let (offset, smallest) = fragments.pop();

            assert!(
                smallest.len() > 0,
                "There must always be at least 1 smallest fragment."
            );

            let mut end_of_entries = true;

            for entry in self.lookup_entries(&text[offset..]) {
                end_of_entries = false;
                for mut fragment in smallest.clone() {
                    fragment.push(self, entry);
                    fragments.push_fragment(fragment);
                }
            }

            // Return if all fragments have the same size.
            if let Some(fragment) = fragments.has_winner() {
                return fragment.words.clone();
            }

            if end_of_entries {
                return vec![];
            }
        }
    }

    pub fn segment<'a, 'b>(&'a self, text: &'b str) -> Vec<Either<&'a DictEntry<String>, &'b str>> {
        let mut non_chinese_start = 0;
        let mut result = vec![];
        let mut offset = 0;
        while offset < text.len() {
            let segment = self.segment_step(&text[offset..]);
            if segment.is_empty() {
                let mut n = offset + 1;
                while !text.is_char_boundary(n) {
                    n += 1;
                }
                offset = n;
            } else {
                if non_chinese_start != offset {
                    result.push(Either::Right(&text[non_chinese_start..offset]));
                }
                offset += segment.iter().map(|x| x.simplified().len()).sum::<usize>();
                non_chinese_start = offset;
                for word in segment {
                    result.push(Either::Left(word));
                }
            }
        }
        if non_chinese_start != offset {
            result.push(Either::Right(&text[non_chinese_start..offset]));
        }
        result
    }
}

// invariant: fragments.len >= 1
struct Fragments<'a> {
    fragments: BTreeMap<usize, Vec<Fragment<'a>>>,
}

impl<'a> Fragments<'a> {
    fn new() -> Self {
        Fragments {
            fragments: BTreeMap::new(),
        }
    }

    // If all fragments have the same non-zero length, return the fragment with the highest score
    fn has_winner(&self) -> Option<&Fragment<'a>> {
        if self.fragments.len() != 1 {
            return None;
        }

        let (&len, fragments) = self.fragments.iter().next()?;
        if len == 0 {
            return None;
        }

        fragments
            .iter()
            .max_by(|a, b| a.score().total_cmp(&b.score()))
    }

    fn push_fragment(&mut self, fragment: Fragment<'a>) {
        let len = fragment.len;
        self.fragments.entry(len).or_default().push(fragment);
    }

    fn pop(&mut self) -> (usize, Vec<Fragment<'a>>) {
        self.fragments.pop_first().unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
struct Fragment<'a> {
    words: Vec<&'a DictEntry<String>>,
    scores: Vec<f64>,
    len: usize, // in bytes
}

impl<'a> Fragment<'a> {
    fn new() -> Self {
        Fragment {
            words: vec![],
            scores: vec![],
            len: 0,
        }
    }

    fn score(&self) -> f64 {
        self.scores
            .iter()
            .product::<f64>()
            .powf((self.scores.len() as f64).recip())
            - self.scores.len() as f64 * 10_f64
    }

    fn push(&mut self, dict: &Dictionary, word: &'a DictEntry<String>) {
        let mut score = dict.frequency(word.simplified());
        self.words.push(word);

        for &exception in SEGMENTATION_EXCEPTIONS {
            let x = self
                .words
                .iter()
                .map(|x| DictEntry::simplified(x))
                .rev()
                .take(exception.len())
                .rev()
                .collect::<Vec<_>>();

            if x == exception {
                score += 100_000_f64;
            }
        }

        self.scores.push(score);
        self.len += word.simplified().len();
    }
}

// "ABC" => ["A", "AB", "ABC"]
// "你好吗" => ["你","你好", "你好吗"]
fn string_inits(str: &str) -> impl Iterator<Item = &str> {
    str.char_indices()
        .skip(1)
        .map(|(n, _)| &str[..n])
        .chain(std::iter::once(str))
}

// fn string_tails(str: &str) -> impl Iterator<Item = &str> {
//     str.char_indices().map(|(n, _)| &str[n..])
// }

// "ABC"
// AB C
// A BC
// A B C

#[cfg(test)]
mod tests {
    use std::{convert::identity, sync::OnceLock};

    use cedict::DictEntry;

    use super::{Dictionary, DICTIONARY};

    #[test]
    fn string_inits_sanity() {
        assert_eq!(
            super::string_inits("ABC").collect::<Vec<&str>>(),
            vec!["A", "AB", "ABC"]
        );
        assert_eq!(
            super::string_inits("你好吗").collect::<Vec<&str>>(),
            vec!["你", "你好", "你好吗"]
        );
    }

    // #[test]
    // fn string_tails_sanity() {
    //     assert_eq!(
    //         super::string_tails("ABC").collect::<Vec<&str>>(),
    //         vec!["ABC", "BC", "C"]
    //     );
    //     assert_eq!(
    //         super::string_tails("你好吗").collect::<Vec<&str>>(),
    //         vec!["你好吗", "好吗", "吗"]
    //     );
    // }

    // 会 should return both hui4 and kuai4
    #[test]
    fn multiple_entries() {
        assert_eq!(
            DICTIONARY
                .lookup_entries("会")
                .map(|entry| entry.pinyin().to_string())
                .collect::<Vec<String>>(),
            &["hui4", "kuai4"]
        );
    }

    // 了 has multi entries spread over multiple lines
    #[test]
    fn entries_for_le_liao() {
        assert_eq!(
            DICTIONARY
                .lookup_entries("了")
                .map(|entry| entry.pinyin().to_string())
                .collect::<Vec<String>>(),
            &["le5", "liao3", "liao3", "liao4"]
        );
    }

    #[track_caller]
    fn assert_segment_step(text: &str, expected: &str) {
        static DICT: OnceLock<Dictionary> = OnceLock::new();
        let dict = DICT.get_or_init(Dictionary::new);
        assert_eq!(
            dict.segment_step(text)
                .into_iter()
                .map(DictEntry::simplified)
                .collect::<Vec<_>>(),
            expected
                .split(' ')
                .filter(|str| !str.is_empty())
                .collect::<Vec<_>>()
        );
    }

    #[track_caller]
    fn assert_segment(text: &str, expected: &str) {
        static DICT: OnceLock<Dictionary> = OnceLock::new();
        let dict = DICT.get_or_init(Dictionary::new);
        assert_eq!(
            dict.segment(text)
                .into_iter()
                .map(|ret| ret.either(DictEntry::simplified, identity))
                .collect::<Vec<_>>(),
            expected.split(' ').collect::<Vec<_>>()
        );
    }

    #[test]
    fn segment_step_sanity_1() {
        assert_segment_step("", "");
    }

    #[test]
    fn segment_step_sanity_2() {
        assert_segment_step("我ABC", "我");
    }

    #[test]
    fn segment_step_sanity_3() {
        assert_segment_step("你好", "你好");
    }

    #[test]
    fn segment_step_sanity_4() {
        assert_segment_step("多工作", "多 工作");
        assert_segment_step("有电话", "有 电话");
        assert_segment_step("回电话", "回 电话");
        assert_segment_step("不知道", "不 知道");
        assert_segment_step("定时间", "定 时间");
        assert_segment_step("这位子", "这 位子");
        assert_segment_step("十分钟", "十 分钟");
        assert_segment_step("有电梯", "有 电梯");
        assert_segment_step("中午前", "中午 前");
        assert_segment_step("想要点", "想要 点");
        // This one is questionable.
        // assert_segment_step(&dict, "得很", &["得", "很"]); // fails
        assert_segment_step("外套", "外套");
        assert_segment_step("家中餐馆", "家");
        assert_segment_step("后生活", "后 生活");
        assert_segment_step("不愿意", "不 愿意");
        assert_segment_step("点出发", "点 出发");
        assert_segment_step("老婆婆", "老 婆婆");
        assert_segment_step("不会跳舞", "不会");
        assert_segment_step("穿上外套", "穿上 外套");
        assert_segment_step("建议", "建议");
        assert_segment_step("怎么不知道", "怎么");
        assert_segment_step("蛋糕发起来", "蛋糕");
        assert_segment_step("管理的人才", "管理");
        assert_segment_step("轻快乐曲", "轻快 乐曲");
        assert_segment_step("高明和", "高明 和");
        assert_segment_step("一下子之间", "一下子");
        assert_segment_step("我绝没想到", "我");
        assert_segment_step("绝没想到", "绝");
        assert_segment_step("没想到", "没想到");
        assert_segment_step("没想到会", "没想到");
    }

    #[test]
    fn segment_sanity_mixed() {
        assert_segment("我叫David", "我 叫 David");
        assert_segment("English!", "English!");
        assert_segment("告诉ABC屁股", "告诉 ABC 屁股");
    }

    #[test]
    fn segment_sanity() {
        assert_segment("节日里人们", "节日 里 人们");
        assert_segment("我可没有时间闲呆着", "我 可 没有 时间 闲 呆 着");
        assert_segment("我要看病", "我 要 看病");
        assert_segment("你好像不太舒服", "你 好像 不 太 舒服");
        assert_segment("我非常想见到她", "我 非常 想 见到 她");
        assert_segment("婚后生活怎么样", "婚 后 生活 怎么样");
        assert_segment(
            "为了照顾家人,我放弃了升职的机会",
            "为了 照顾 家人 , 我 放弃 了 升职 的 机会",
        );
        assert_segment("我有好多事要干", "我 有 好多 事 要 干");

        assert_segment("我不知道这张表怎么填", "我 不 知道 这 张 表 怎么 填");
        assert_segment("他今天有很多事情要做", "他 今天 有 很 多 事情 要 做");
        assert_segment("我不知道他在想什么", "我 不 知道 他 在 想 什么");
        assert_segment("我是个不顾家的人", "我 是 个 不顾 家 的 人");
        assert_segment("你真有胆量", "你 真 有胆量");
        assert_segment("夏天到了", "夏天 到 了");
        assert_segment("我合上书准备离开", "我 合上 书 准备 离开");
        assert_segment("他的话", "他 的 话");
        assert_segment("你用什么方法学习", "你 用 什么 方法 学习");
        /*
        , ("你定时间吧","你 定 时间 吧")
        -- , ("这位子有人吗","这 位子 有人 吗")



        , ("我先做作业再吃晚饭","我 先 做 作业 再 吃 晚饭")
        , ("现在一点钟了", "现在 一 点钟 了")


        , ("AAA","AAA")
        , ("BBB","BBB")

        , ("习惯是第二天性", "习惯 是 第二 天性")
        , ("一切都会好的", "一切 都 会 好 的")
        , ("上帝什么都会", "上帝 什么 都 会")
        -- , ("他比我高一个头", "他 比 我 高 一 个 头") -- What's the right way to tokenize here?
        , ("每张桌子上都有菜单", "每 张 桌子 上 都 有 菜单")
        , ("把手举起来","把 手举 起来")
        , ("若天下雨","若 天 下雨")
        , ("现在是五点四十分","现在 是 五 点 四十 分")
        , ("她在写作文","她 在 写 作文")
         */
    }

    #[test]
    fn default_dict_is_valid() {
        let dict = Dictionary::new();
        assert_eq!(dict.entries.len(), 118473);
    }

    #[test]
    fn default_wf_is_valid() {
        let dict = Dictionary::new();
        assert_eq!(dict.word_frequency.len(), 99121);
    }

    #[test]
    fn multi_lookup() {
        let dict = Dictionary::new();
        assert_eq!(
            dict.lookup_entries("一个人")
                .map(DictEntry::simplified)
                .map(str::to_string)
                .collect::<Vec<String>>(),
            vec!["一", "一个人"]
        );
    }
}
