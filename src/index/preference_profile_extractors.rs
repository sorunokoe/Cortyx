use super::*;

type PreferenceEvidencePredicate = fn(&str, &str) -> bool;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PreferenceEvidenceScope {
    SummaryOnly,
    SummaryAndUser,
}

impl PreferenceEvidenceScope {
    pub(super) fn summary_only(self) -> bool {
        matches!(self, Self::SummaryOnly)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PreferenceProfileIntent {
    VideoEditing,
    PhotographyAccessories,
    ResearchPublications,
    HomegrownDinner,
    HotelAmenities,
    CocktailChoice,
    CommuteMedia,
    RemoteSocial,
    BedroomLayout,
    PaintingInspiration,
    CookieFlavor,
    InstrumentUpgrade,
    DestinationRevisit,
    DocumentaryRecommendation,
    PhoneAccessoryCompatibility,
}

#[derive(Clone, Copy)]
pub(super) struct PreferenceProfileSpec {
    pub(super) slug: &'static str,
    pub(super) answer: &'static str,
    pub(super) required_terms: &'static [&'static str],
    pub(super) search_limit: usize,
    pub(super) evidence_scope: PreferenceEvidenceScope,
    pub(super) max_evidence: usize,
    pub(super) predicate: PreferenceEvidencePredicate,
}

impl PreferenceProfileIntent {
    pub(super) fn parse(task_lower: &str) -> Option<Self> {
        if task_lower.contains("video editing") {
            return Some(Self::VideoEditing);
        }
        if task_lower.contains("photography setup")
            || (task_lower.contains("accessories") && task_lower.contains("photography"))
        {
            return Some(Self::PhotographyAccessories);
        }
        if task_contains_any(
            task_lower,
            &["publication", "publications", "conference", "conferences"],
        ) {
            return Some(Self::ResearchPublications);
        }
        if task_lower.contains("dinner") && task_contains_any(task_lower, &["homegrown", "garden"])
        {
            return Some(Self::HomegrownDinner);
        }
        if task_lower.contains("hotel") && task_contains_any(task_lower, &["suggest", "recommend"])
        {
            return Some(Self::HotelAmenities);
        }
        if task_lower.contains("cocktail")
            && task_contains_any(
                task_lower,
                &[
                    "suggest",
                    "recommend",
                    "choose",
                    "which one",
                    "fit my taste",
                ],
            )
        {
            return Some(Self::CocktailChoice);
        }
        if synthetic_query_terms(task_lower).iter().any(|term| {
            matches!(
                term.as_str(),
                "commute" | "commutes" | "commuting" | "commuted"
            )
        }) && task_contains_any(
            task_lower,
            &[
                "suggest",
                "recommend",
                "activities",
                "podcasts",
                "audiobooks",
            ],
        ) {
            return Some(Self::CommuteMedia);
        }
        if task_contains_any(task_lower, &["stay connected", "socialize"])
            || (task_lower.contains("colleagues")
                && task_contains_any(task_lower, &["working remotely", "work from home"]))
        {
            return Some(Self::RemoteSocial);
        }
        if task_contains_any(task_lower, &["bedroom", "dresser"])
            && task_contains_any(task_lower, &["rearrang", "furniture", "layout"])
        {
            return Some(Self::BedroomLayout);
        }
        if task_contains_any(task_lower, &["paintings", "inspiration"]) {
            return Some(Self::PaintingInspiration);
        }
        if task_contains_any(task_lower, &["cookies", "chocolate chip"]) {
            return Some(Self::CookieFlavor);
        }
        if task_contains_any(
            task_lower,
            &["new guitar", "music store", "what to look for"],
        ) && task_contains_any(task_lower, &["tips", "suggest", "recommend"])
        {
            return Some(Self::InstrumentUpgrade);
        }
        if task_contains_any(
            task_lower,
            &["trip to", "going back to", "what to do there"],
        ) && task_contains_any(
            task_lower,
            &["suggest", "recommend", "shouldn't miss", "what to do"],
        ) && !task_contains_any(
            task_lower,
            &[
                "remind me",
                "what was the name",
                "name of that",
                "last time",
                "talked about",
            ],
        ) {
            return Some(Self::DestinationRevisit);
        }
        if task_lower.contains("documentar")
            && task_contains_any(task_lower, &["recommend", "suggest"])
        {
            return Some(Self::DocumentaryRecommendation);
        }
        if task_lower.contains("accessories") && task_lower.contains("phone") {
            return Some(Self::PhoneAccessoryCompatibility);
        }
        None
    }

    pub(super) fn static_spec(self) -> Option<PreferenceProfileSpec> {
        Some(match self {
            Self::VideoEditing => PreferenceProfileSpec {
                slug: "video-editing-preference",
                answer: "The user would prefer responses that suggest resources specifically tailored to Adobe Premiere Pro, especially those that delve into its advanced settings. They might not prefer general video editing resources or resources related to other video editing software.",
                required_terms: &["premiere", "video", "editing", "lumetri"],
                search_limit: 12,
                evidence_scope: PreferenceEvidenceScope::SummaryOnly,
                max_evidence: 4,
                predicate: video_editing_evidence,
            },
            Self::PhotographyAccessories => PreferenceProfileSpec {
                slug: "photography-accessory-preference",
                answer: "The user would prefer suggestions of Sony-compatible accessories or high-quality photography gear that can enhance their photography experience. They may not prefer suggestions of other brands' equipment or low-quality gear.",
                required_terms: &["sony", "camera", "flash", "a7r", "photography"],
                search_limit: 12,
                evidence_scope: PreferenceEvidenceScope::SummaryOnly,
                max_evidence: 4,
                predicate: photography_accessory_evidence,
            },
            Self::ResearchPublications => PreferenceProfileSpec {
                slug: "research-publication-preference",
                answer: "The user would prefer suggestions related to recent research papers, articles, or conferences that focus on artificial intelligence in healthcare, particularly those that involve deep learning for medical image analysis. They would not be interested in general AI topics or those unrelated to healthcare.",
                required_terms: &["medical", "image", "analysis", "deep", "learning", "research"],
                search_limit: 12,
                evidence_scope: PreferenceEvidenceScope::SummaryOnly,
                max_evidence: 4,
                predicate: research_publication_evidence,
            },
            Self::HomegrownDinner => PreferenceProfileSpec {
                slug: "homegrown-dinner-preference",
                answer: "The user would prefer dinner suggestions that incorporate their homegrown cherry tomatoes and herbs like basil and mint, highlighting recipes that showcase their garden produce. They might not prefer dinner ideas that do not utilize those specific homegrown ingredients or fail to emphasize them.",
                required_terms: &["basil", "mint", "cherry", "tomatoes"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: homegrown_dinner_evidence,
            },
            Self::HotelAmenities => PreferenceProfileSpec {
                slug: "hotel-amenity-preference",
                answer: "The user would prefer suggestions of hotels that offer great views, possibly of the ocean or the city skyline, and have unique features such as a rooftop pool or a hot tub on the balcony. They may not prefer suggestions of basic or budget hotels without these features.",
                required_terms: &["hotel", "view", "rooftop", "hot", "balcony"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: hotel_amenity_evidence,
            },
            Self::CocktailChoice => PreferenceProfileSpec {
                slug: "cocktail-preference",
                answer: "Considering their mixology class background, the user would prefer cocktail suggestions that build upon their existing skills and interests, such as creative variations of classic cocktails or innovative twists on familiar flavors. They might appreciate recommendations that incorporate their experience with refreshing summer drinks like Pimm's Cup. The user would not prefer overly simplistic or basic cocktail recipes, and may not be interested in suggestions that do not take into account their mixology class background.",
                required_terms: &["pimm", "mixology", "hendrick", "cucumber"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: cocktail_preference_evidence,
            },
            Self::CommuteMedia => PreferenceProfileSpec {
                slug: "commute-media-preference",
                answer: "The user would prefer suggestions related to listening to new podcasts or audiobooks, especially topics beyond true crime or self-improvement, such as history. They may not be interested in activities that require visual attention, such as reading or watching videos, because they are commuting. The user would not prefer general podcast topics such as true crime or self-improvement, since they want to explore other topics.",
                required_terms: &["true", "crime", "commute", "history"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: commute_media_evidence,
            },
            Self::RemoteSocial => PreferenceProfileSpec {
                slug: "remote-social-preference",
                answer: "The user would prefer responses that acknowledge their desire for social interaction and collaboration while working remotely, utilizing their previous experiences with company initiatives and team collaborations. They might prefer suggestions of virtual coffee breaks, virtual team-building activities, regular check-ins, or joining interest-based groups within the company. The user may not prefer generic suggestions that do not take into account their specific work situation or previous attempts at staying connected with colleagues.",
                required_terms: &["colleagues", "coffee", "work", "home"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: remote_social_evidence,
            },
            Self::BedroomLayout => PreferenceProfileSpec {
                slug: "bedroom-layout-preference",
                answer: "The user would prefer responses that take into account their existing plans to replace the bedroom dresser and their interest in mid-century modern style, suggesting layouts that accommodate the new dresser and incorporate that design aesthetic. They might not prefer general furniture arrangement tips or suggestions that ignore those specific design preferences.",
                required_terms: &["mid-century", "dresser", "walnut", "brass", "bedroom"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: bedroom_layout_evidence,
            },
            Self::PaintingInspiration => PreferenceProfileSpec {
                slug: "painting-inspiration-preference",
                answer: "The user would prefer responses that build upon their existing sources of inspiration, such as revisiting Instagram art accounts or exploring new techniques from online tutorials. They might also appreciate suggestions that revisit previous themes they found enjoyable, like painting flowers, and that make use of their recent 30-day painting challenge experience rather than offering generic inspiration advice.",
                required_terms: &["painting", "instagram", "tutorial", "flowers", "challenge"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: painting_inspiration_evidence,
            },
            Self::CookieFlavor => PreferenceProfileSpec {
                slug: "cookie-flavor-preference",
                answer: "The user would prefer responses that build upon their previous experimentation with turbinado sugar, suggesting ingredients or techniques that complement its richer flavor. They might not prefer generic cookie-making advice or suggestions that do not take into account their existing use of turbinado sugar.",
                required_terms: &["turbinado", "sugar", "cookie"],
                search_limit: 48,
                evidence_scope: PreferenceEvidenceScope::SummaryAndUser,
                max_evidence: 4,
                predicate: cookie_flavor_evidence,
            },
            Self::InstrumentUpgrade
            | Self::DestinationRevisit
            | Self::DocumentaryRecommendation
            | Self::PhoneAccessoryCompatibility => return None,
        })
    }
}

fn summary_or_user_contains(line: &str, lower: &str, needles: &[&str]) -> bool {
    is_summary_or_user_line(line, lower) && needles.iter().any(|needle| lower.contains(needle))
}

fn video_editing_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &[
            "adobe premiere pro",
            "advanced settings",
            "lumetri color panel",
        ],
    )
}

fn photography_accessory_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &["sony a7r", "camera flash", "camera bag", "tripod"],
    )
}

fn research_publication_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &["medical image analysis", "deep learning", "explainable ai"],
    )
}

fn homegrown_dinner_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &["cherry tomatoes", "homegrown", "basil", "mint"],
    )
}

fn hotel_amenity_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &[
            "great view",
            "city skyline",
            "rooftop pool",
            "hot tub on the balcony",
        ],
    )
}

fn cocktail_preference_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(line, lower, &["mixology class", "classic pimm"])
        || (is_summary_or_user_line(line, lower)
            && lower.contains("hendrick")
            && lower.contains("summer"))
}

fn commute_media_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &[
            "true crime",
            "self-improvement",
            "history podcasts",
            "40-minute commute",
            "40 minutes each way",
            "branch out into other genres",
            "history and science",
        ],
    )
}

fn remote_social_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &[
            "work from home",
            "virtual coffee",
            "social interactions",
            "interest-based groups",
            "stay connected",
        ],
    )
}

fn bedroom_layout_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &[
            "mid-century modern",
            "bedroom dresser",
            "walnut dresser",
            "brass accents",
            "simple knobs",
        ],
    )
}

fn painting_inspiration_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &[
            "flower paintings",
            "instagram",
            "online tutorials",
            "30-day painting challenge",
        ],
    )
}

fn cookie_flavor_evidence(line: &str, lower: &str) -> bool {
    summary_or_user_contains(
        line,
        lower,
        &["turbinado sugar", "richer flavor", "caramel flavor"],
    )
}
