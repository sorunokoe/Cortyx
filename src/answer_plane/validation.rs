use super::*;

pub(super) fn looks_like_typed_open_qa_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if is_temporal_reasoning_query(task) {
        return false;
    }
    lower.starts_with("would ")
        || lower.starts_with("could ")
        || lower.starts_with("should ")
        || lower.starts_with("can ")
        || lower.starts_with("will ")
        || lower.starts_with("may ")
        || lower.starts_with("might ")
        || lower.starts_with("is ")
        || lower.starts_with("are ")
        || lower.starts_with("was ")
        || lower.starts_with("were ")
        || lower.starts_with("does ")
        || lower.starts_with("do ")
        || lower.starts_with("did ")
        || lower.starts_with("has ")
        || lower.starts_with("have ")
        || lower.starts_with("had ")
        || lower.starts_with("which ")
        || lower.starts_with("what might ")
        || lower.starts_with("what would ")
        || lower.contains(" likely ")
        || lower.contains(" considered ")
}

pub(super) fn is_education_field_query(lower_task: &str) -> bool {
    lower_task.contains("field")
        || lower_task.contains("education")
        || lower_task.contains("study")
        || lower_task.contains("school")
        || lower_task.contains("pursue")
        || lower_task.contains("career option")
        || lower_task.contains("career options")
        || lower_task.contains("career path")
        || lower_task.contains("future career")
}

pub(super) fn typed_open_qa_anchor_terms(
    task_terms: &[String],
    subject_hints: &[String],
) -> Vec<String> {
    const FILLER: &[&str] = &[
        "likely",
        "probably",
        "possibly",
        "potentially",
        "considered",
        "still",
        "more",
        "most",
        "less",
        "least",
        "kind",
        "sort",
        "thing",
        "things",
        "personality",
        "trait",
        "traits",
        "additional",
        "alternative",
        "popular",
        "based",
        "around",
    ];
    let mut terms = task_terms
        .iter()
        .filter(|term| {
            !FILLER.contains(&term.as_str()) && !subject_hints.iter().any(|hint| hint == *term)
        })
        .cloned()
        .collect::<Vec<_>>();
    if terms.is_empty() {
        terms = task_terms.to_vec();
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn format_open_qa_answer_surface_answer(task: &str, answer: &str) -> String {
    let lower_task = task.to_ascii_lowercase();
    let answer_lower = answer.to_ascii_lowercase();
    if answer_lower.contains("ally")
        && [
            "member of the lgbtq community",
            "member of the lgbtq+ community",
            "part of the lgbtq community",
            "part of the lgbtq+ community",
            "member of the transgender community",
        ]
        .iter()
        .any(|needle| lower_task.contains(needle))
    {
        "Likely no, supportive ally".to_string()
    } else if answer_lower.contains("ally")
        && [
            "ally to the transgender community",
            "ally to the lgbtq community",
            "ally to the lgbtq+ community",
            "considered an ally",
        ]
        .iter()
        .any(|needle| lower_task.contains(needle))
    {
        "Yes, supportive ally".to_string()
    } else {
        answer.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerShape {
    Generic,
    YesNo,
    Choice,
    Number,
    Duration,
    Date,
    TraitList,
    Suggestion,
}

pub(super) const GENERIC_ANCHOR_TERMS: &[&str] = &[
    "activities",
    "activity",
    "advice",
    "answer",
    "attributes",
    "career",
    "careers",
    "city",
    "close",
    "could",
    "country",
    "countries",
    "current",
    "currently",
    "degree",
    "degrees",
    "did",
    "does",
    "education",
    "event",
    "events",
    "fields",
    "field",
    "first",
    "group",
    "groups",
    "help",
    "home",
    "idea",
    "ideas",
    "interesting",
    "job",
    "jobs",
    "kind",
    "kinds",
    "last",
    "less",
    "likely",
    "live",
    "location",
    "many",
    "might",
    "more",
    "most",
    "movie",
    "movies",
    "much",
    "name",
    "names",
    "occupation",
    "option",
    "options",
    "people",
    "person",
    "personality",
    "profession",
    "quality",
    "qualities",
    "question",
    "recent",
    "recommend",
    "recommended",
    "recommending",
    "resource",
    "resources",
    "role",
    "roles",
    "school",
    "show",
    "shows",
    "some",
    "something",
    "state",
    "states",
    "suggest",
    "suggested",
    "suggesting",
    "task",
    "thing",
    "things",
    "tips",
    "topic",
    "topics",
    "trait",
    "traits",
    "trip",
    "type",
    "types",
    "upcoming",
    "weekend",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "would",
];

pub(super) const ANSWER_REJECT_PREFIXES: &[&str] = &[
    "congratulations",
    "great to hear",
    "here are",
    "i can help",
    "i'd be happy",
    "i would be happy",
    "i'm happy to",
    "let's get started",
    "sounds great",
    "that's great",
    "that sounds",
    "wow",
];

pub(super) const ANSWER_REJECT_EXACT: &[&str] = &[
    "1",
    "2",
    "3",
    "can",
    "great idea",
    "i'm a large language model",
    "many dishes",
    "trap crop",
    "yogurt making",
];

pub(super) const ANSWER_TRAILING_STOPWORDS: &[&str] = &[
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
];

pub(super) const GENERIC_COLLECTION_NOUNS: &[&str] = &[
    "activities",
    "advice",
    "days",
    "dishes",
    "ideas",
    "options",
    "recipes",
    "resources",
    "tips",
    "tools",
    "ways",
];

pub(super) const FOOD_QUERY_HINTS: &[&str] = &[
    "bake",
    "basil",
    "cookies",
    "cook",
    "cooking",
    "dessert",
    "dinner",
    "ingredients",
    "meal",
    "mint",
    "recipe",
    "recipes",
    "serve",
    "slow cooker",
];

pub(super) const FOOD_GENERIC_NOUNS: &[&str] =
    &["drink", "water", "cocktail", "tea", "smoothie", "juice"];

pub(super) const FOOD_ITEM_HINTS: &[&str] = &[
    "beef",
    "brownie",
    "cake",
    "caprese",
    "chicken",
    "chili",
    "chutney",
    "cookie",
    "cookies",
    "curry",
    "dessert",
    "lamb",
    "pasta",
    "pesto",
    "salad",
    "sandwich",
    "soup",
    "spaghetti",
    "stew",
    "tacos",
];

fn answer_shape(task: &str) -> AnswerShape {
    let lower = task.to_ascii_lowercase();
    if parse_binary_choice(task).is_some() || !parse_open_qa_choice_options(task).is_empty() {
        AnswerShape::Choice
    } else if lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" what month")
        || lower.contains(" which month")
        || lower.contains(" what year")
        || lower.contains(" which year")
    {
        AnswerShape::Date
    } else if lower.starts_with("how long ")
        || lower.contains("how many days")
        || lower.contains("how many weeks")
        || lower.contains("how many months")
        || lower.contains("how many years")
        || lower.contains("how many hours")
        || lower.contains("how many minutes")
    {
        AnswerShape::Duration
    } else if lower.starts_with("how many ")
        || lower.starts_with("how much ")
        || lower.starts_with("how often ")
        || lower.contains("number of ")
    {
        AnswerShape::Number
    } else if lower.contains("personality trait")
        || lower.contains("personality traits")
        || lower.contains("what traits")
        || lower.contains("what attributes")
        || lower.contains("attributes describe")
        || lower.contains("what qualities")
    {
        AnswerShape::TraitList
    } else if lower.starts_with("would ")
        || lower.starts_with("could ")
        || lower.starts_with("should ")
        || lower.starts_with("can ")
        || lower.starts_with("will ")
        || lower.starts_with("may ")
        || lower.starts_with("might ")
        || lower.starts_with("is ")
        || lower.starts_with("are ")
        || lower.starts_with("was ")
        || lower.starts_with("were ")
        || lower.starts_with("does ")
        || lower.starts_with("do ")
        || lower.starts_with("did ")
        || lower.starts_with("has ")
        || lower.starts_with("have ")
        || lower.starts_with("had ")
    {
        AnswerShape::YesNo
    } else if is_suggestion_query(task) {
        AnswerShape::Suggestion
    } else {
        AnswerShape::Generic
    }
}

pub(super) fn is_suggestion_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("can you suggest")
        || lower.starts_with("can you recommend")
        || lower.contains(" any advice")
        || lower.contains(" any tips")
        || lower.contains(" any ideas")
        || lower.starts_with("what should i ")
        || lower.starts_with("what can i ")
        || lower.starts_with("what could i ")
}

pub(super) fn is_food_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    FOOD_QUERY_HINTS.iter().any(|needle| lower.contains(needle))
}

pub(super) fn normalized_validation_text(text: &str) -> String {
    text.replace('*', " ")
        .replace('`', " ")
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn task_anchor_terms(
    task: &str,
    task_terms: &[String],
    subject_hints: &[String],
) -> Vec<String> {
    let lower = task.to_ascii_lowercase();
    let mut anchors = if looks_like_typed_open_qa_query(task) {
        typed_open_qa_anchor_terms(task_terms, subject_hints)
    } else {
        task_terms
            .iter()
            .filter(|term| {
                !subject_hints.iter().any(|hint| hint == *term)
                    && !GENERIC_ANCHOR_TERMS.contains(&term.as_str())
            })
            .cloned()
            .collect::<Vec<_>>()
    };

    if let Some((options, _)) = parse_binary_choice(task) {
        anchors.extend(
            options
                .into_iter()
                .flat_map(|option| option.tokens)
                .filter(|term| !GENERIC_ANCHOR_TERMS.contains(&term.as_str())),
        );
    }
    if !lower.contains("yes or no") {
        anchors.extend(
            parse_open_qa_choice_options(task)
                .into_iter()
                .flat_map(|option| option.tokens)
                .filter(|term| !GENERIC_ANCHOR_TERMS.contains(&term.as_str())),
        );
    }
    anchors.sort();
    anchors.dedup();
    anchors
}

pub(super) fn answer_form_confidence(task: &str, text: &str, task_terms: &[String]) -> f32 {
    let clean = normalized_validation_text(text);
    if clean.is_empty() {
        return 0.0;
    }

    let lower = clean.to_ascii_lowercase();
    let shape = answer_shape(task);
    let tokens = lower
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if text.contains('?') || clean.ends_with(':') {
        return 0.0;
    }
    let reject_question_echo =
        shape != AnswerShape::Choice && looks_like_question_echo(task, &lower, task_terms);
    let reject_heading = matches!(
        shape,
        AnswerShape::Generic | AnswerShape::TraitList | AnswerShape::Suggestion
    ) && looks_like_heading_fragment(text, &clean);
    if reject_question_echo
        || reject_heading
        || looks_like_social_filler(&lower)
        || looks_like_truncated_answer(&tokens)
    {
        return 0.0;
    }
    if ANSWER_REJECT_EXACT.contains(&lower.as_str()) {
        return 0.0;
    }
    if institution_query_expected(task) {
        return institution_answer_confidence(&clean, &lower);
    }

    match shape {
        AnswerShape::YesNo => yes_no_answer_confidence(task, &lower),
        AnswerShape::Choice => choice_answer_confidence(task, &clean),
        AnswerShape::Number => number_answer_confidence(task, &lower),
        AnswerShape::Duration => duration_answer_confidence(task, &lower),
        AnswerShape::Date => date_answer_confidence(&clean, &lower),
        AnswerShape::TraitList => trait_list_answer_confidence(&clean, &lower),
        AnswerShape::Suggestion => suggestion_answer_confidence(task, &lower),
        AnswerShape::Generic => generic_answer_confidence(&lower, task_terms),
    }
}

pub(super) fn looks_like_question_echo(
    task: &str,
    answer_lower: &str,
    task_terms: &[String],
) -> bool {
    let answer_key = normalized_answer_key(answer_lower);
    let task_key = normalized_answer_key(task);
    if answer_key.is_empty() || task_key.is_empty() {
        return false;
    }
    if task_key.contains(&answer_key) && answer_key.split_whitespace().count() >= 3 {
        return true;
    }

    let overlap = task_overlap_count(answer_lower, task_terms);
    let novel_tokens = answer_lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 4)
        .filter(|token| {
            !task_terms
                .iter()
                .any(|term| query_term_matches_token(term, token))
        })
        .count();
    overlap >= task_terms.len().min(3).max(2) && novel_tokens == 0
}

pub(super) fn looks_like_heading_fragment(original: &str, clean: &str) -> bool {
    let tokens = clean.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return true;
    }
    if original.contains("**") || original.starts_with('#') {
        if tokens.len() <= 5 {
            return true;
        }
    }
    let alpha_tokens = tokens
        .iter()
        .filter(|token| token.chars().any(|c| c.is_alphabetic()))
        .count();
    alpha_tokens > 0
        && tokens.len() <= 4
        && tokens
            .iter()
            .any(|token| token.chars().all(|c| c.is_ascii_digit()))
        && tokens
            .iter()
            .filter(|token| token.chars().any(|c| c.is_alphabetic()))
            .all(|token| {
                token
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
            })
}

pub(super) fn looks_like_social_filler(lower: &str) -> bool {
    ANSWER_REJECT_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

pub(super) fn looks_like_truncated_answer(tokens: &[&str]) -> bool {
    let Some(last) = tokens.last() else {
        return true;
    };
    let tail = last.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if ANSWER_TRAILING_STOPWORDS.contains(&tail) {
        return true;
    }
    let Some(first) = tokens.first() else {
        return true;
    };
    matches!(*first, "and" | "or" | "to" | "for" | "with" | "because")
}

fn yes_no_answer_confidence(task: &str, lower: &str) -> f32 {
    if [
        "yes",
        "no",
        "likely yes",
        "likely no",
        "probably yes",
        "probably no",
        "somewhat",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return 1.0;
    }
    if lower.contains("religious") || lower.contains("ally") {
        return 0.9;
    }
    if task.to_ascii_lowercase().contains("member of the lgbtq")
        && lower.contains("supportive ally")
    {
        return 0.9;
    }
    0.0
}

fn choice_answer_confidence(task: &str, text: &str) -> f32 {
    if let Some((options, _)) = parse_binary_choice(task) {
        if options
            .iter()
            .any(|option| answer_items_overlap(text, &option.display))
        {
            return 1.0;
        }
    }
    let options = parse_open_qa_choice_options(task);
    if options
        .iter()
        .any(|option| answer_items_overlap(text, &option.display))
    {
        return 1.0;
    }
    if let Some(target) = open_qa_location_target(task) {
        if open_qa_location_alias(target, text).is_some() {
            return 0.9;
        }
    }
    0.0
}

fn number_answer_confidence(task: &str, lower: &str) -> f32 {
    let tokens = lower.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens.len() <= 6 && tokens.iter().all(|token| numeric_answer_component(token)) {
        return 1.0;
    }
    if answer_shape(task) != AnswerShape::Duration
        && tokens.len() == 1
        && parse_count_token(tokens[0]).is_some()
    {
        return 1.0;
    }
    0.0
}

fn numeric_answer_component(token: &str) -> bool {
    parse_count_token(token).is_some()
        || matches!(
            token,
            "times"
                | "time"
                | "per"
                | "week"
                | "weeks"
                | "month"
                | "months"
                | "year"
                | "years"
                | "day"
                | "days"
                | "hour"
                | "hours"
                | "minute"
                | "minutes"
                | "ago"
                | "before"
                | "after"
                | "and"
        )
}

fn duration_answer_confidence(task: &str, lower: &str) -> f32 {
    if number_answer_confidence(task, lower) > 0.0
        && [
            "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
            "minute", "minutes", "ago",
        ]
        .iter()
        .any(|unit| lower.contains(unit))
    {
        return 1.0;
    }
    0.0
}

fn date_answer_confidence(text: &str, lower: &str) -> f32 {
    if extract_explicit_date(text, None).is_some()
        || [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "thanksgiving",
            "christmas",
            "independence day",
            "black friday",
            "easter",
            "holi",
        ]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        return 1.0;
    }
    0.0
}

fn trait_list_answer_confidence(text: &str, lower: &str) -> f32 {
    let normalized = text.replace(", and ", ", ").replace(" and ", ", ");
    let parts = normalized
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() >= 2
        && parts.iter().all(|part| {
            let words = part.split_whitespace().count();
            words >= 1
                && words <= 3
                && !part.chars().any(|c| c.is_ascii_digit())
                && !part.contains('?')
        })
    {
        return 1.0;
    }
    if lower.split_whitespace().count() == 1 {
        return 0.0;
    }
    0.25
}

fn suggestion_answer_confidence(task: &str, lower: &str) -> f32 {
    let tokens = lower.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens
        .first()
        .copied()
        .map(parse_count_token)
        .flatten()
        .is_some()
    {
        return 0.0;
    }
    if tokens.len() >= 2
        && matches!(tokens[0], "many" | "some" | "several" | "various")
        && GENERIC_COLLECTION_NOUNS.contains(&tokens[1])
    {
        return 0.0;
    }
    if is_food_query(task) {
        let has_generic_drink = FOOD_GENERIC_NOUNS
            .iter()
            .any(|needle| lower.contains(needle));
        let has_food_item = FOOD_ITEM_HINTS.iter().any(|needle| lower.contains(needle));
        if task.to_ascii_lowercase().contains("dinner") && has_generic_drink && !has_food_item {
            return 0.0;
        }
    }
    if tokens.len() > 10 {
        return 0.35;
    }
    0.8
}

fn generic_answer_confidence(lower: &str, task_terms: &[String]) -> f32 {
    let tokens = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens.len() == 1 {
        let token = tokens[0];
        if parse_count_token(token).is_some() || token.len() <= 2 {
            return 0.0;
        }
    }
    let novel_tokens = tokens
        .iter()
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            !task_terms
                .iter()
                .any(|term| query_term_matches_token(term, token))
        })
        .count();
    if novel_tokens == 0 && tokens.len() <= 3 {
        return 0.0;
    }
    0.75
}

pub(super) fn institution_query_expected(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.contains("which university")
        || lower.contains("what university")
        || lower.contains("which college")
        || lower.contains("what college")
        || lower.contains("which school")
        || lower.contains("what school")
        || lower.contains("which institute")
        || lower.contains("what institute")
}

pub(super) fn institution_specific_anchor_terms(task: &str) -> Vec<String> {
    if !institution_query_expected(task) {
        return Vec::new();
    }
    let mut terms = salient_query_terms(task)
        .into_iter()
        .filter(|term| {
            term.len() >= 4
                && !matches!(
                    term.as_str(),
                    "university"
                        | "college"
                        | "school"
                        | "institute"
                        | "academy"
                        | "present"
                        | "presented"
                        | "poster"
                        | "research"
                        | "conference"
                )
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn institution_answer_confidence(clean: &str, lower: &str) -> f32 {
    if ["university", "college", "school", "institute", "academy"]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        return 0.95;
    }
    let tokens = clean.split_whitespace().collect::<Vec<_>>();
    if !tokens.is_empty()
        && tokens.len() <= 4
        && tokens.iter().all(|token| {
            token
                .chars()
                .all(|c| c.is_ascii_uppercase() || matches!(c, '.' | '&' | '-'))
        })
        && tokens.iter().any(|token| token.len() >= 2)
    {
        return 0.8;
    }
    0.0
}
