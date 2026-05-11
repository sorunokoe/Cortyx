use super::*;

pub(super) fn generate_answer_surface_rows(text: &str) -> Vec<AnswerSurfaceRow> {
    const JOB_PATTERN: &str = "job occupation profession work career role";
    const LOCATION_PATTERN: &str = "live location residence city home moved based";
    const DEGREE_PATTERN: &str = "degree major education field studied graduated";
    const PET_PATTERN: &str = "pet cat dog name called";
    const FAMILY_TRIP_PATTERN: &str = "family trip vacation destination travel location";
    const ISSUE_PATTERN: &str = "issue problem malfunction wrong service repair not functioning";
    const VEHICLE_PATTERN: &str = "vehicle car model current vehicle";
    const PRODUCT_PATTERN: &str = "current product brand shampoo conditioner skincare use";
    const SHOE_BRAND_PATTERN: &str = "favorite running shoes brand shoe sneaker trainer";
    const CERTIFICATION_PATTERN: &str = "certification credential completed last month recent";
    const GIFT_PATTERN: &str = "birthday gift sister present bought";
    const PLAY_PATTERN: &str = "play theater community theater attended watched";
    const CONCERT_VENUE_PATTERN: &str = "concert venue attended live show";
    const RICE_PATTERN: &str = "favorite rice type grain";
    const INSTAGRAM_FOLLOWERS_PATTERN: &str =
        "instagram followers follower count current social media";
    const PRE_1920_COIN_PATTERN: &str = "pre-1920 coins collection count total";
    const NATIONAL_GEOGRAPHIC_PATTERN: &str = "national geographic issues finished reading count";
    const KOREAN_RESTAURANT_PATTERN: &str = "korean restaurants tried city count";
    const FISH_CATCH_PATTERN: &str = "largemouth bass fishing trip catch count";
    const PLAYLIST_PATTERN: &str = "playlist music spotify called name";
    const GROUP_PATTERN: &str = "kind type group joined online group community";
    const SIGN_PATTERN: &str = "sign warning notice precaution precautionary cafe café";
    const RELAX_ACTIVITY_PATTERN: &str = "relax unwind nature walk hike road trip activity";
    const RESEARCH_PATTERN: &str = "research researched topic investigating looking into";
    const FITNESS_RECORD_PATTERN: &str =
        "personal best time record fastest race run charity 5k score";

    let mut rows = Vec::new();
    for raw_line in text.split(['\n', '.', '!', '?']) {
        let line = raw_line.trim();
        if line.len() < 10 {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let mut push = |question_pattern: &str, answer_span: Option<String>, confidence: f32| {
            push_answer_surface_row_for_line(
                &mut rows,
                line,
                question_pattern,
                answer_span,
                confidence,
            );
        };

        push(
            JOB_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "i work as ",
                    "i'm a ",
                    "i am a ",
                    "i'm an ",
                    "i am an ",
                    "i work as an ",
                    "i work as a ",
                    "i became a ",
                    "i became an ",
                ],
                &[
                    " at ",
                    " for ",
                    " in ",
                    " with ",
                    " because ",
                    " since ",
                    " and ",
                    " but ",
                ],
                4,
            )
            .filter(|value| looks_like_job_surface_value(value)),
            0.92,
        );

        push(
            LOCATION_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "i live in ",
                    "i'm living in ",
                    "i am living in ",
                    "i moved to ",
                    "i moved back to ",
                    "i relocated to ",
                    "i settled in ",
                    "i'm based in ",
                    "i am based in ",
                ],
                &[
                    " with ",
                    " because ",
                    " and ",
                    " but ",
                    " now ",
                    " again ",
                    " after ",
                    " so ",
                ],
                4,
            ),
            0.91,
        );

        push(
            DEGREE_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "degree in ",
                    "majored in ",
                    "major in ",
                    "graduated with a degree in ",
                    "graduated in ",
                    "studied ",
                ],
                &[" at ", " from ", " and ", " but ", " because "],
                4,
            ),
            0.9,
        );

        push(
            PET_PATTERN,
            extract_fact_after_any(
                line,
                &lower,
                &[
                    "my cat's name is ",
                    "my dog's name is ",
                    "my cat is named ",
                    "my dog is named ",
                    "our cat's name is ",
                    "our dog's name is ",
                ],
                &[" and ", " but ", " because "],
                2,
            ),
            0.95,
        );

        push(ISSUE_PATTERN, extract_issue_surface_value(line), 0.84);
        push(
            RESEARCH_PATTERN,
            extract_research_surface_value(line, &lower),
            0.84,
        );
        push(
            FITNESS_RECORD_PATTERN,
            extract_fitness_record_surface_value(line, &lower),
            0.9,
        );

        if lower.contains("as a family") || lower.contains("with my family") {
            push(
                FAMILY_TRIP_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "thinking of going to ",
                        "planning a trip to ",
                        "planned a trip to ",
                        "went to ",
                    ],
                    &[" with ", " for ", " and ", " but ", " because "],
                    4,
                ),
                0.82,
            );
        }

        if lower.contains("joined a ") || lower.contains("joined an ") || lower.contains("joined ")
        {
            push(
                GROUP_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &["joined a ", "joined an ", "joined "],
                    &[" last ", " and ", " but ", " because ", " to ", " with "],
                    5,
                ),
                0.83,
            );
        }

        if lower.contains("model") || lower.contains("vehicle") || lower.contains("car") {
            push(
                VEHICLE_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "switched to a ",
                        "switched to an ",
                        "bought a ",
                        "bought an ",
                        "drive a ",
                        "drive an ",
                    ],
                    &[" model", " because ", " and ", " but "],
                    4,
                ),
                0.8,
            );
        }

        if lower.contains("using") || lower.contains("shampoo") || lower.contains("conditioner") {
            push(
                PRODUCT_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "i switched to using ",
                        "i use ",
                        "i'm using ",
                        "i am using ",
                        "i switched to ",
                    ],
                    &[" for ", " because ", " and ", " but "],
                    4,
                ),
                0.78,
            );
            push(
                PRODUCT_PATTERN,
                extract_shampoo_brand_surface_value(line, &lower),
                0.86,
            );
        }

        push(
            SHOE_BRAND_PATTERN,
            extract_running_shoe_brand_surface_value(line, &lower),
            0.87,
        );
        push(
            CERTIFICATION_PATTERN,
            extract_certification_surface_value(line, &lower),
            0.88,
        );
        push(
            GIFT_PATTERN,
            extract_sister_gift_surface_value(line, &lower),
            0.84,
        );
        push(
            PLAY_PATTERN,
            extract_theater_play_surface_value(line, &lower),
            0.84,
        );
        push(
            CONCERT_VENUE_PATTERN,
            extract_concert_venue_surface_value(line, &lower),
            0.84,
        );
        push(
            RICE_PATTERN,
            extract_favorite_rice_surface_value(line, &lower),
            0.84,
        );

        if let Some((question_pattern, value)) = extract_relative_location_surface_row(line, &lower)
        {
            push(&question_pattern, Some(value), 0.86);
        }

        push(
            INSTAGRAM_FOLLOWERS_PATTERN,
            extract_instagram_followers_surface_value(line, &lower),
            0.86,
        );
        push(
            PRE_1920_COIN_PATTERN,
            extract_pre_1920_coin_surface_value(line, &lower),
            0.84,
        );
        push(
            NATIONAL_GEOGRAPHIC_PATTERN,
            extract_national_geographic_count_surface_value(line, &lower),
            0.82,
        );
        push(
            KOREAN_RESTAURANT_PATTERN,
            extract_korean_restaurant_count_surface_value(line, &lower),
            0.82,
        );
        push(
            FISH_CATCH_PATTERN,
            extract_largemouth_bass_count_surface_value(line, &lower),
            0.82,
        );

        if lower.contains("playlist") {
            push(
                PLAYLIST_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "playlist called ",
                        "playlist is called ",
                        "named my playlist ",
                    ],
                    &[" and ", " but ", " because "],
                    4,
                ),
                0.84,
            );
        }

        if lower.contains("sign ") {
            push(
                SIGN_PATTERN,
                extract_fact_after_any(
                    line,
                    &lower,
                    &[
                        "sign saying ",
                        "sign said ",
                        "sign that said ",
                        "sign reading ",
                        "sign read ",
                    ],
                    &[" and ", " but ", " because ", " near ", " at "],
                    8,
                ),
                0.8,
            );
        }

        push(
            RELAX_ACTIVITY_PATTERN,
            extract_relax_activity_surface_value(line, &lower),
            0.76,
        );
    }
    rows
}

fn push_answer_surface_row_for_line(
    rows: &mut Vec<AnswerSurfaceRow>,
    line: &str,
    question_pattern: &str,
    answer_span: Option<String>,
    confidence: f32,
) {
    let Some(answer_span) = answer_span else {
        return;
    };

    push_answer_surface_row(
        rows,
        question_pattern,
        Some(answer_span.clone()),
        confidence,
    );

    let Some(scoped_pattern) =
        scoped_question_pattern(question_pattern, dialogue_line_scope_speaker(line))
    else {
        return;
    };
    push_answer_surface_row(
        rows,
        &scoped_pattern,
        Some(answer_span),
        (confidence + 0.03).min(0.95),
    );
}

fn dialogue_line_scope_speaker(line: &str) -> Option<&str> {
    let (speaker, rest) = line.split_once(':')?;
    if !is_dialogue_speaker(speaker) {
        return None;
    }
    let rest = rest.trim();
    let lower = rest.to_ascii_lowercase();
    let ellided_self_reference = lower.starts_with("researching ")
        || lower.starts_with("looking into ")
        || lower.starts_with("working in ")
        || lower.starts_with("working on ")
        || lower.starts_with("planning ")
        || lower.starts_with("hoping ")
        || lower.starts_with("trying ");
    (lower.starts_with("i ")
        || lower.starts_with("i'")
        || lower.starts_with("i’m")
        || lower.starts_with("my ")
        || lower.starts_with("we ")
        || lower.starts_with("our ")
        || ellided_self_reference)
        .then_some(speaker.trim())
}
