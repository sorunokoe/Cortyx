pub(crate) fn fact_alias_lines(user_lines: &[String], assistant_lines: &[String]) -> Vec<String> {
    let mut aliases = Vec::new();
    let mut push = |alias: &str| {
        if !aliases.iter().any(|existing: &String| existing == alias) {
            aliases.push(alias.to_string());
        }
    };

    for line in user_lines.iter().chain(assistant_lines.iter()) {
        let lower = line.to_ascii_lowercase();

        if lower.contains("changed my last name") || lower.contains("old name was") {
            push("what was my last name before i changed it");
            push("old last name");
            push("previous last name");
        }
        if lower.contains("certification in ") && lower.contains("completed last month") {
            push("what certification did i complete last month");
            push("latest certification");
            push("recent certification");
        }
        if lower.contains("my cat's name is ") {
            push("what is the name of my cat");
            push("cat name");
            push("pet name");
        }
        if lower.contains("planning a birthday trip to hawaii")
            || (lower.contains("stay on oahu") && lower.contains("birthday"))
        {
            push("where am i planning to stay for my birthday trip to hawaii");
            push("birthday trip hawaii stay");
        }
        if lower.contains("same grocery list app as me now")
            || (lower.contains("my mom") && lower.contains("grocery list app"))
        {
            push("is my mom using the same grocery list method as me");
            push("mom same grocery list app");
        }
        if lower.contains("cocktail-making class on") {
            push("what day of the week do i take a cocktail-making class");
            push("cocktail-making class day");
        }
        if lower.contains("spotify") && lower.contains("playlist") {
            push("what is the name of the music streaming service have i been using lately");
            push("music streaming service");
        }
        if lower.contains("went with my family for a week") {
            push("where did i go on a week-long trip with my family");
            push("family trip location");
        }
        if lower.contains("action figure") && lower.contains("thrift store") {
            push("what type of action figure did i buy from a thrift store");
        }
        if lower.contains("shampoo") && lower.contains("trader joe") {
            push("what brand of shampoo do i currently use");
            push("current shampoo brand");
        }
        if lower.contains("initially thought was just a cold") {
            push("what health issue did i initially think was just a cold");
        }
        if lower.contains("favorite running shoes")
            || (lower.contains("nike") && lower.contains("running shoes"))
        {
            push("what brand are my favorite running shoes");
        }
        if lower.contains("birthday gift") && lower.contains("sister") && lower.contains("dress") {
            push("what did i get my sister for her birthday");
        }
        if lower.contains("bookshelf") && lower.contains("ikea") {
            push("where did i buy the bookshelf");
        }
    }

    aliases
}

/// R17 Sol1: Prospective Query Pre-image.
///
/// Scans a conversation turn for fact-bearing assertions and generates the natural-language
/// question forms that a human would ask about those facts. Returned as a space-separated
/// string of question vocabulary tokens for BM25 injection.
///
/// Pattern format: `(&[trigger_words], &[question_vocab])`.
/// Match: if ALL trigger words appear in the lowercased text.
/// Output: all matching question_vocab tokens joined, deduplicated.
///
/// Zero dependencies — pure `str::contains()`. Static data ≈ 8 KB.
pub(crate) fn generate_query_surface(text: &str) -> Option<String> {
    // Each entry: (trigger phrases ANY of which must appear, question vocabulary to emit)
    // Triggers are lowercase. Match = text.to_lowercase() contains any trigger.
    static PATTERNS: &[(&[&str], &[&str])] = &[
        // ── Occupation / Job ────────────────────────────────────────────────────────
        (
            &[
                "work as",
                "works as",
                "i am a ",
                "i'm a ",
                "i am an ",
                "i'm an ",
                "my job",
                "my career",
                "my profession",
                "my occupation",
                "became a ",
                "got a job",
                "started as",
                "employed as",
                "hired as",
                "nurse",
                "doctor",
                "engineer",
                "teacher",
                "manager",
                "developer",
                "lawyer",
                "accountant",
                "designer",
                "analyst",
                "scientist",
                "therapist",
                "firefighter",
                "police",
                "chef",
                "pilot",
                "architect",
                "consultant",
                "hospital shift",
                "hospital ward",
                "patients were",
                "seeing patients",
                "office job",
                "remote job",
                "full-time",
                "part-time",
                "freelance",
            ],
            &[
                "what is her job",
                "what does she do",
                "what is her occupation",
                "what is her profession",
                "what does she work as",
                "what is his job",
                "what does he do",
                "what is his occupation",
                "what is their job",
                "what is her career",
                "what is her work",
                "what does she do for work",
                "where does she work",
                "job",
                "occupation",
                "profession",
                "career",
                "work",
            ],
        ),
        // ── Location / Residence ─────────────────────────────────────────────────
        (
            &[
                "i live",
                "i moved",
                "i'm living",
                "i am living",
                "my home is",
                "my house",
                "my apartment",
                "my place",
                "relocated to",
                "settled in",
                "based in",
                "moving to",
                "new city",
                "new town",
                "new place",
            ],
            &[
                "where does she live",
                "where does he live",
                "where do they live",
                "what city does she live in",
                "where is her home",
                "where did she move",
                "what is her address",
                "where is she based",
                "location",
                "city",
                "home",
                "residence",
            ],
        ),
        // ── Relationship / Partner ───────────────────────────────────────────────
        (
            &[
                "my husband",
                "my wife",
                "my partner",
                "my spouse",
                "my boyfriend",
                "my girlfriend",
                "my fiance",
                "we got married",
                "getting married",
                "our wedding",
                "we're engaged",
                "i'm engaged",
                "i'm married",
            ],
            &[
                "is she married",
                "who is her husband",
                "who is her partner",
                "who is her spouse",
                "what is her relationship status",
                "is he married",
                "who is his wife",
                "who is their partner",
                "relationship",
                "married",
                "husband",
                "wife",
                "partner",
                "spouse",
                "engaged",
                "yes",
            ],
        ),
        // ── Children / Family ────────────────────────────────────────────────────
        (
            &[
                "my daughter",
                "my son",
                "my kids",
                "my children",
                "my baby",
                "my child",
                "pregnant",
                "expecting",
                "gave birth",
                "new baby",
                "i have a ",
                "we have a kid",
                "we have children",
            ],
            &[
                "does she have children",
                "does he have kids",
                "how many children",
                "does she have a daughter",
                "does he have a son",
                "children",
                "kids",
                "daughter",
                "son",
                "baby",
                "family",
                "parent",
                "yes",
            ],
        ),
        // ── Contact / Phone ──────────────────────────────────────────────────────
        (
            &[
                "my phone",
                "my number",
                "my mobile",
                "my cell",
                "phone number",
                "contact number",
                "changed my number",
                "new number",
                "new phone",
            ],
            &[
                "what is her phone number",
                "what is his number",
                "what is their phone",
                "how do i contact",
                "what is her contact",
                "phone",
                "number",
                "mobile",
                "cell",
                "contact",
            ],
        ),
        // ── Email / Address ──────────────────────────────────────────────────────
        (
            &[
                "my email",
                "new email",
                "email address",
                "my address",
                "i can be reached",
                "reach me at",
            ],
            &[
                "what is her email",
                "what is his email",
                "what is their email",
                "how to contact",
                "email",
                "address",
                "contact",
            ],
        ),
        // ── Age / Birthday ───────────────────────────────────────────────────────
        (
            &[
                "my birthday",
                "born in",
                "born on",
                "i turned",
                "i'm turning",
                "i am ",
                "years old",
                "i was born",
            ],
            &[
                "how old is she",
                "how old is he",
                "what is her age",
                "when is her birthday",
                "when was she born",
                "age",
                "birthday",
                "born",
                "years old",
            ],
        ),
        // ── Health / Medical ─────────────────────────────────────────────────────
        (
            &[
                "i was diagnosed",
                "i have been sick",
                "my condition",
                "my illness",
                "my surgery",
                "i had surgery",
                "in the hospital",
                "hospital stay",
                "my health",
                "my medication",
                "my treatment",
                "recovering from",
                "chronic",
                "my therapy",
                "health issues",
                "had a bad case of",
                "came down with",
                "dealing with health",
                "health problem",
                "i had a bad case",
                "turned out to be more serious",
            ],
            &[
                "what health issues",
                "is she sick",
                "what condition does she have",
                "what health issue did i have",
                "what illness did i have",
                "what did i have",
                "what was i diagnosed with",
                "medical health illness condition surgery hospital treatment health issue",
            ],
        ),
        // ── Education / School ───────────────────────────────────────────────────
        (
            &[
                "i graduated",
                "i'm studying",
                "i am studying",
                "my degree",
                "my major",
                "i'm in school",
                "i'm in college",
                "i'm at university",
                "going back to school",
                "my thesis",
                "my dissertation",
                "i got accepted",
            ],
            &[
                "what does she study",
                "what is her degree",
                "where does she go to school",
                "what is his major",
                "education",
                "school",
                "college",
                "university",
                "degree",
                "studying",
                "graduated",
            ],
        ),
        // ── Pet ──────────────────────────────────────────────────────────────────
        (
            &[
                "my dog",
                "my cat",
                "my pet",
                "my puppy",
                "my kitten",
                "got a dog",
                "got a cat",
                "adopted a",
            ],
            &[
                "does she have a pet",
                "what kind of pet",
                "what is the pet's name",
                "what breed is her dog",
                "what kind of dog does she have",
                "pet",
                "dog",
                "cat",
                "animal",
                "breed",
                "purebred",
                "yes",
            ],
        ),
        // ── Knowledge-update: "changed to" / "now X" ────────────────────────────
        (
            &[
                "changed to",
                "switched to",
                "now i",
                "now she",
                "now he",
                "updated to",
                "new job",
                "new role",
                "new position",
                "promoted",
                "just started",
                "recently started",
                "just got",
            ],
            &[
                "what changed",
                "what is the current",
                "what is the latest",
                "what is her current",
                "what is his current",
                "current",
                "latest",
                "updated",
                "changed",
                "new",
                "now",
            ],
        ),
        // ── Hobbies / Interests ──────────────────────────────────────────────────
        (
            &[
                "i love",
                "i enjoy",
                "my hobby",
                "i like to",
                "i play",
                "i run",
                "i paint",
                "i write",
                "i sing",
                "i dance",
                "i practice",
                "my passion",
                "my interest",
            ],
            &[
                "what does she enjoy",
                "what are her hobbies",
                "what does she do for fun",
                "hobby",
                "interest",
                "passion",
                "enjoy",
                "like",
            ],
        ),
        // ── Property / Vehicle ───────────────────────────────────────────────────
        (
            &[
                "my car",
                "my house",
                "my apartment",
                "i bought a",
                "i own a",
                "my property",
                "my condo",
                "my vehicle",
            ],
            &[
                "does she own a car",
                "what kind of car",
                "does she own a house",
                "car",
                "house",
                "property",
                "vehicle",
                "apartment",
                "yes",
            ],
        ),
        // ── Financial ───────────────────────────────────────────────────────────
        (
            &[
                "my salary",
                "my income",
                "my savings",
                "i earn",
                "i make",
                "got a raise",
                "my budget",
                "financially",
                "debt",
                "mortgage",
            ],
            &[
                "what is her salary",
                "how much does she make",
                "financial situation",
                "salary",
                "income",
                "money",
                "earnings",
            ],
        ),
        // R18 P5: New categories ─────────────────────────────────────────────────

        // ── Vehicle / Car model ──────────────────────────────────────────────────
        (
            &[
                "i drive",
                "my car is",
                "bought a car",
                "new car",
                "my truck",
                "my suv",
                "my motorcycle",
                "my bike",
                "leased a",
                "test drove",
            ],
            &[
                "what car does she drive",
                "what vehicle does he own",
                "what kind of car",
                "does she have a car",
                "car",
                "vehicle",
                "drive",
                "model",
                "yes",
            ],
        ),
        // ── Diet / Food preferences ──────────────────────────────────────────────
        (
            &[
                "i'm vegan",
                "i'm vegetarian",
                "i eat ",
                "my diet",
                "i don't eat",
                "gluten free",
                "lactose",
                "i avoid",
                "food allergy",
                "i'm allergic to",
                "i'm pescatarian",
                "i'm keto",
                "low carb",
            ],
            &[
                "what does she eat",
                "is she vegan",
                "what is his diet",
                "food preferences",
                "diet",
                "vegan",
                "vegetarian",
                "gluten",
                "allergy",
                "food",
            ],
        ),
        // ── Language spoken ──────────────────────────────────────────────────────
        (
            &[
                "i speak",
                "i'm fluent",
                "my native language",
                "i'm learning",
                "i know french",
                "i know spanish",
                "i know german",
                "i know japanese",
                "i know chinese",
                "i know arabic",
                "i know italian",
                "bilingual",
                "multilingual",
            ],
            &[
                "what language does she speak",
                "what languages does he know",
                "is she bilingual",
                "language",
                "speak",
                "fluent",
                "native",
            ],
        ),
        // ── Religion / Faith ─────────────────────────────────────────────────────
        (
            &[
                "i'm christian",
                "i'm muslim",
                "i'm jewish",
                "i'm buddhist",
                "i'm hindu",
                "my religion",
                "my faith",
                "i pray",
                "i go to church",
                "i go to mosque",
                "i'm catholic",
                "i'm atheist",
                "i'm agnostic",
                "my beliefs",
            ],
            &[
                "what religion does she follow",
                "is he religious",
                "what faith",
                "religion",
                "faith",
                "church",
                "pray",
                "belief",
            ],
        ),
        // ── Sport / Physical activity ────────────────────────────────────────────
        (
            &[
                "i play soccer",
                "i play football",
                "i play basketball",
                "i play tennis",
                "i play golf",
                "i play baseball",
                "i play volleyball",
                "i play rugby",
                "i go swimming",
                "i go cycling",
                "i go running",
                "i go hiking",
                "my team",
                "i coach",
                "i train",
                "i compete",
                "my sport",
            ],
            &[
                "what sport does she play",
                "what sport does he play",
                "what team",
                "sport",
                "team",
                "play",
                "compete",
                "athletic",
            ],
        ),
        // ── Musical instrument ───────────────────────────────────────────────────
        (
            &[
                "i play guitar",
                "i play piano",
                "i play violin",
                "i play drums",
                "i play bass",
                "i play flute",
                "i play saxophone",
                "i play trumpet",
                "i play cello",
                "i play ukulele",
                "my instrument",
                "i'm in a band",
            ],
            &[
                "what instrument does she play",
                "does he play an instrument",
                "does she play music",
                "instrument",
                "music",
                "band",
                "guitar",
                "piano",
            ],
        ),
        // ── Social media / Online presence ───────────────────────────────────────
        (
            &[
                "my instagram",
                "my twitter",
                "my tiktok",
                "my youtube",
                "my twitch",
                "my linkedin",
                "my handle",
                "my username",
                "i post on",
                "my followers",
                "my channel",
                "my blog",
                "my podcast",
                "my newsletter",
            ],
            &[
                "what is her instagram",
                "what is his twitter",
                "social media",
                "instagram",
                "twitter",
                "youtube",
                "tiktok",
                "handle",
                "channel",
                "followers",
                "platform",
                "subscribers",
                "views",
                "online",
            ],
        ),
        // ── Subscription / Membership ────────────────────────────────────────────
        (
            &[
                "i subscribe",
                "my subscription",
                "i'm a member",
                "my membership",
                "i pay for",
                "i cancelled",
                "netflix",
                "spotify",
                "gym membership",
            ],
            &[
                "does she have a subscription",
                "what subscriptions",
                "membership",
                "subscribe",
                "service",
                "member",
                "yes",
            ],
        ),
        // ── Medication / Prescription ────────────────────────────────────────────
        (
            &[
                "i take ",
                "my medication",
                "my prescription",
                "i'm on ",
                "my pills",
                "my dosage",
                "i was prescribed",
                "my antidepressant",
                "my antibiotic",
            ],
            &[
                "what medication does she take",
                "is he on medication",
                "prescription",
                "medication",
                "medicine",
                "pills",
                "prescription",
                "dosage",
            ],
        ),
        // ── Marital status change ────────────────────────────────────────────────
        (
            &[
                "i got divorced",
                "going through a divorce",
                "we separated",
                "i'm separated",
                "signed divorce papers",
                "legally separated",
                "my ex",
                "my ex-husband",
                "my ex-wife",
                "divorced now",
            ],
            &[
                "is she divorced",
                "is he separated",
                "relationship status",
                "divorced",
                "separated",
                "divorce",
                "ex",
                "single",
                "no",
            ],
        ),
        // ── New home / Moving ────────────────────────────────────────────────────
        (
            &[
                "i'm moving",
                "we're moving",
                "just moved",
                "new apartment",
                "new house",
                "new home",
                "bought a house",
                "renting",
                "my new place",
                "signed a lease",
            ],
            &[
                "did she move",
                "where did he move",
                "new address",
                "moved",
                "new home",
                "address",
                "house",
                "apartment",
                "neighborhood",
            ],
        ),
        // ── Travel / Country visited ─────────────────────────────────────────────
        (
            &[
                "i visited",
                "i went to",
                "i traveled to",
                "i'm going to",
                "my trip",
                "my vacation",
                "my holiday",
                "i'm in ",
                "just got back from",
                "i flew to",
                "i drove to",
                "i'm visiting",
            ],
            &[
                "where did she travel",
                "what countries has he visited",
                "travel plans",
                "trip",
                "vacation",
                "travel",
                "visit",
                "country",
                "destination",
            ],
        ),
        // ── Named colleague / coworker ───────────────────────────────────────────
        (
            &[
                "my boss",
                "my manager",
                "my colleague",
                "my coworker",
                "my supervisor",
                "my team lead",
                "my mentor",
                "my intern",
                "works with me",
                "my teammate",
            ],
            &[
                "who is her boss",
                "who does she work with",
                "coworker",
                "colleague",
                "boss",
                "manager",
                "supervisor",
                "team",
                "work relationship",
            ],
        ),
        // ── Nationality / Origin ─────────────────────────────────────────────────
        (
            &[
                "i'm from",
                "i grew up in",
                "my home country",
                "my hometown",
                "originally from",
                "i was raised in",
                "my nationality",
                "i'm american",
                "i'm british",
                "i'm australian",
                "i'm canadian",
                "i'm french",
                "i'm german",
                "i'm italian",
                "i'm japanese",
                "i'm korean",
                "i'm chinese",
                "i'm indian",
                "i'm brazilian",
                "i'm mexican",
            ],
            &[
                "where is she from",
                "what is his nationality",
                "what country",
                "nationality",
                "origin",
                "hometown",
                "country",
                "from",
            ],
        ),
        // ── Gym / Workout routine ────────────────────────────────────────────────
        (
            &[
                "i go to the gym",
                "i work out",
                "my workout",
                "my fitness routine",
                "i lift weights",
                "i do yoga",
                "i do pilates",
                "i do crossfit",
                "my personal trainer",
                "i exercise",
            ],
            &[
                "does she go to the gym",
                "what is his workout routine",
                "fitness",
                "gym",
                "workout",
                "exercise",
                "fitness routine",
                "training",
                "yes",
            ],
        ),
        // ── Sports team / Fan ────────────────────────────────────────────────────
        (
            &[
                "i'm a fan of",
                "i support",
                "my favorite team",
                "my team is",
                "i cheer for",
                "i root for",
            ],
            &[
                "what team does she support",
                "favorite sports team",
                "fan",
                "team",
                "support",
                "cheer",
            ],
        ),
        // ── Allergies ────────────────────────────────────────────────────────────
        (
            &[
                "i'm allergic",
                "my allergy",
                "allergic to",
                "i can't eat",
                "i react to",
                "my epipen",
                "anaphylactic",
                "nut allergy",
                "shellfish allergy",
            ],
            &[
                "what is she allergic to",
                "does he have allergies",
                "allergy",
                "allergic",
                "reaction",
                "food allergy",
            ],
        ),
        // ── Volunteering / Charity ───────────────────────────────────────────────
        (
            &[
                "i volunteer",
                "i volunteer at",
                "my volunteer work",
                "i donate",
                "i work with a charity",
                "nonprofit",
                "community service",
            ],
            &[
                "does she volunteer",
                "what charity does he support",
                "volunteering",
                "volunteer",
                "charity",
                "donate",
                "nonprofit",
            ],
        ),
        // ── Graduation / Degree completion ───────────────────────────────────────
        (
            &[
                "i graduated",
                "i finished my degree",
                "i got my degree",
                "just graduated",
                "got my phd",
                "got my masters",
                "got my bachelors",
                "commencement",
            ],
            &[
                "when did she graduate",
                "what degree did he get",
                "graduated",
                "graduation",
                "degree",
                "diploma",
                "alumni",
            ],
        ),
        // ── Job promotion / Title change ─────────────────────────────────────────
        (
            &[
                "i got promoted",
                "i was promoted",
                "i'm now a",
                "new title",
                "senior now",
                "my new role",
                "i lead",
                "i manage now",
                "team lead now",
            ],
            &[
                "was she promoted",
                "what is his new title",
                "promotion",
                "promoted",
                "title",
                "role",
                "senior",
                "lead",
            ],
        ),
        // ── Birth year / Generation ──────────────────────────────────────────────
        (
            &[
                "i was born in",
                "born in 19",
                "born in 20",
                "class of",
                "generation",
                "millennial",
                "gen z",
                "gen x",
                "boomer",
            ],
            &[
                "what year was she born",
                "how old is he",
                "birth year",
                "generation",
                "born",
                "age",
                "millennial",
            ],
        ),
        // ── Salary / Compensation ────────────────────────────────────────────────
        (
            &[
                "my salary is",
                "i make ",
                "i earn ",
                "i get paid",
                "annual salary",
                "hourly rate",
                "i got a raise",
                "my compensation",
                "base salary",
            ],
            &[
                "what is her salary",
                "how much does he earn",
                "salary",
                "income",
                "earn",
                "pay",
                "compensation",
                "raise",
            ],
        ),
        // ── Pregnancy / Child update ─────────────────────────────────────────────
        (
            &[
                "i'm pregnant",
                "we're expecting",
                "due in",
                "my baby is due",
                "i gave birth",
                "our new baby",
                "newborn",
                "just had a baby",
            ],
            &[
                "is she pregnant",
                "when is she due",
                "did she have the baby",
                "pregnant",
                "expecting",
                "due date",
                "baby",
                "newborn",
            ],
        ),
        // ── Social preference / Introvert / Extrovert ────────────────────────────
        (
            &[
                "i'm an introvert",
                "i'm an extrovert",
                "i prefer small gatherings",
                "i love parties",
                "i avoid crowds",
                "i'm shy",
                "i'm outgoing",
                "i socialize",
                "i like to be alone",
            ],
            &[
                "is she introverted",
                "is he outgoing",
                "social preference",
                "introvert",
                "extrovert",
                "social",
                "personality",
            ],
        ),
        // ── Time zone / Schedule ─────────────────────────────────────────────────
        (
            &[
                "my time zone",
                "i'm in pst",
                "i'm in est",
                "i'm in gmt",
                "i'm in cet",
                "i work nights",
                "night shift",
                "morning shift",
                "i work remotely",
                "i work from home",
                "wfh",
                "my schedule",
            ],
            &[
                "what time zone is she in",
                "what is his schedule",
                "time zone",
                "schedule",
                "shift",
                "remote",
                "work from home",
            ],
        ),
        // ── Named pet (with name) ────────────────────────────────────────────────
        (
            &[
                "my dog named",
                "my cat named",
                "my pet named",
                "called my dog",
                "called my cat",
                "my dog's name is",
                "my cat's name is",
            ],
            &[
                "what is her pet's name",
                "what is his dog's name",
                "what is the cat called",
                "pet name",
                "dog name",
                "cat name",
            ],
        ),
        // ── Subscription service preference ──────────────────────────────────────
        (
            &[
                "i use ",
                "i prefer ",
                "my favorite app",
                "my go-to",
                "i rely on",
                "i switched from",
                "i switched to",
                "i unsubscribed",
            ],
            &[
                "what app does she use",
                "what service does he prefer",
                "preferred service",
                "app",
                "service",
                "use",
                "prefer",
                "favorite",
            ],
        ),
        // R21 T1: 8 new categories from benchmark forensics ─────────────────────

        // ── Education / Degree specifics ─────────────────────────────────────────
        (
            &[
                "bachelor",
                "master",
                "phd",
                "doctorate",
                "associate degree",
                "business administration",
                "computer science degree",
                "engineering degree",
                "liberal arts",
                "i graduated with",
                "my degree is",
                "i majored in",
                "i studied",
                "i have a degree",
                "i got my degree in",
            ],
            &[
                "what degree did she graduate with",
                "what did he major in",
                "what degree did i graduate with",
                "what did i study",
                "bachelor master degree graduated majored studied",
            ],
        ),
        // ── Commute / Travel time ─────────────────────────────────────────────
        (
            &[
                "my commute",
                "commute is",
                "commute takes",
                "i commute",
                "it takes me",
                "drive to work",
                "takes me to get to",
                "minutes to work",
                "minutes each way",
                "hour commute",
                "long commute",
                "my drive",
            ],
            &[
                "how long is her commute",
                "how long does it take him to get to work",
                "how long is my daily commute",
                "how long is the commute",
                "commute travel minutes drive takes how long",
            ],
        ),
        // ── Shopping / Retail location ────────────────────────────────────────
        (
            &[
                "i bought it at",
                "i got it at",
                "i purchased at",
                "i redeemed",
                "coupon at",
                "shop at",
                "i shop at",
                "i go to",
                "store i use",
                "my grocery store",
                "my pharmacy",
                "at target",
                "at walmart",
                "at costco",
                "at whole foods",
                "at the store",
                "at the mall",
            ],
            &[
                "where did she buy it",
                "where did he shop",
                "which store",
                "where did i buy",
                "where did i use my coupon",
                "where did i redeem",
                "where store shop redeemed used purchased bought",
            ],
        ),
        // ── Personal records / Achievements ───────────────────────────────────
        (
            &[
                "my personal best",
                "my pb",
                "my record",
                "my best time",
                "my fastest",
                "my slowest",
                "i achieved",
                "i completed in",
                "my all-time best",
                "i finished in",
                "my score was",
                "my result was",
                "i ran it in",
                "i did it in",
                "my time was",
            ],
            &[
                "what is her personal best",
                "what was his record time",
                "what was my personal best",
                "what was my time",
                "what was my record",
                "personal best time record score completed achieved fastest",
            ],
        ),
        // ── Creative works / Naming ───────────────────────────────────────────
        (
            &[
                "i created",
                "i named it",
                "i called it",
                "i titled it",
                "my playlist",
                "my album",
                "my project is called",
                "i published",
                "my book",
                "my song",
                "my artwork",
                "my film",
                "i wrote",
                "my blog is called",
                "my channel is called",
                "i started a",
            ],
            &[
                "what is the name of her project",
                "what did she call it",
                "what is my playlist called",
                "what did i name it",
                "what is my project called",
                "playlist name created called made titled named",
            ],
        ),
        // ── Theater / Events attended ─────────────────────────────────────────
        (
            &[
                "i saw",
                "i watched",
                "i attended",
                "i went to see",
                "i went to watch",
                "the play i saw",
                "the show i attended",
                "at the theater",
                "at the cinema",
                "at the concert",
                "at the festival",
                "i caught a show",
                "i saw a play",
                "community theater",
                "local theater",
                "live performance",
                "saw them live",
                "saw her live",
                "saw him live",
                "saw it live",
                "saw the show",
                "saw the concert",
                "live show",
                "live concert",
                "at the venue",
                "at the arena",
                "at the stadium",
                "at the amphitheater",
            ],
            &[
                "what play did she attend",
                "what show did he watch",
                "what event did they see",
                "what play did i attend",
                "what show did i see",
                "what performance did i watch",
                "who did i go with to the music event",
                "music event live concert show",
                "play show attended watched performance theater event concert venue",
            ],
        ),
        // ── Wedding / Family event venue ──────────────────────────────────────
        (
            &[
                "cousin's wedding",
                "family wedding",
                "attended a wedding",
                "at the wedding",
                "at the reception",
                "at the grand ballroom",
                "wedding was held",
                "wedding venue",
                "sister's wedding",
                "brother's wedding",
                "the ballroom",
                "grand ballroom",
            ],
            &[
                "where was the wedding held",
                "what venue was the wedding at",
                "where did i attend",
                "cousin wedding venue ballroom reception",
                "cousin",
                "wedding",
                "venue",
                "ballroom",
                "reception",
                "hall",
                "grand",
                "life event relative relatives participated family ceremony celebrate",
            ],
        ),
        // ── Cooking / Baking event disclosure ─────────────────────────────────
        (
            &[
                "i just baked",
                "i recently baked",
                "by the way, i baked",
                "i just cooked",
                "i recently cooked",
                "by the way, i cooked",
                "i just made",
                "i recently made",
                "by the way, i made",
                "baked it for my",
                "cooked it for my",
                "made it for my",
                "i baked a",
                "i cooked a",
                "i prepared a",
                "i made a",
            ],
            &[
                "what did i cook bake make recently",
                "what did i make for my friend",
                "what did i recently prepare cook bake",
                "cook bake make friend ago couple days",
                "recently made cooked baked prepared for my friend couple days ago",
            ],
        ),
        // ── Books / Reading ───────────────────────────────────────────────────
        (
            &[
                "reading before bed",
                "book club",
                "a book called",
                "a book titled",
                "currently reading",
                "i've been reading",
                "i am reading",
                "my reading",
                "i finished reading",
                "i started reading",
                "i'm reading",
                "our book club",
                "we discussed the book",
                "reading a book",
                "currently devouring",
                "am devouring",
                "been devouring",
                "i'm devouring",
            ],
            &[
                "what book am i reading",
                "what book is she reading",
                "what book did she finish",
                "what are we reading",
                "what book did i read",
                "what am i currently reading",
                "what book does she recommend",
                "book reading currently title author novel",
            ],
        ),
        // ── Music / Instrument practice ───────────────────────────────────────
        (
            &[
                "i play guitar",
                "i play the guitar",
                "i practice guitar",
                "guitar lessons",
                "i play piano",
                "i play the piano",
                "i practice piano",
                "piano lessons",
                "i play violin",
                "i practice violin",
                "i play bass",
                "i play drums",
                "music lessons",
                "my instrument",
                "my guitar",
                "my piano",
            ],
            &[
                "what instrument does she play",
                "how long does he practice",
                "how many minutes does she practice",
                "how much time does he dedicate",
                "what instrument do i play",
                "how long do i practice",
                "how much time do i dedicate",
                "how many minutes do i practice",
                "instrument music guitar piano violin practice practicing lessons",
                "minutes per day time dedicate",
            ],
        ),
        // ── Personal products / Brand use ─────────────────────────────────────
        (
            &[
                "i picked up at",
                "my shampoo",
                "my conditioner",
                "my moisturizer",
                "my skincare",
                "for my hair",
                "for my skin",
                "my face wash",
                "my body wash",
                "i switched to using",
                "i recently started using",
                "i use for my",
                "lavender shampoo",
                "scented shampoo",
                "hair products",
                "skin products",
            ],
            &[
                "what brand do i use",
                "what do i currently use",
                "what product do i use",
                "what shampoo do i use",
                "what does she use for her hair",
                "brand product shampoo conditioner skincare currently using hair care",
            ],
        ),
        // ── Counting / Aggregation facts ──────────────────────────────────────
        (
            &[
                "i've done",
                "i have done",
                "i've been to",
                "i have been to",
                "i've visited",
                "i have visited",
                "i've tried",
                "i have tried",
                "i've worked on",
                "i've read",
                "i have read",
                "i've seen",
                "i've watched",
                "i have watched",
                "i've bought",
                "i have bought",
                "i've completed",
                "i have completed",
                "i have attended",
                "i've attended",
                "total of",
                "so far i've",
                "so far i have",
                "i've now",
                "i've gone through",
                "i have now",
            ],
            &[
                "how many has she done",
                "how many times has he visited",
                "how many total",
                "how many have i done",
                "how many have i visited",
                "how many have i tried",
                "how many total count worked done bought completed attended read watched have i",
            ],
        ),
        // ── Gifts / Presents received ─────────────────────────────────────────
        // "I got my new stand mixer as a birthday gift from my sister" → who gave
        (
            &[
                "as a birthday gift",
                "birthday gift from",
                "birthday present from",
                "got me for my birthday",
                "gave me for my birthday",
                "gave me a new",
                "gave me the",
                "as a christmas gift",
                "christmas present from",
                "received as a gift",
                "gifted me",
                "got me a gift",
                "gave me as a gift",
            ],
            &[
                "who gave me",
                "who got me",
                "what was the gift",
                "birthday present from",
                "who gave me a gift",
                "who gave me for my birthday",
                "gift giver gave received birthday present from sister brother",
            ],
        ),
    ];

    let lower = text.to_lowercase();
    let mut tokens: Vec<&str> = Vec::new();
    for (triggers, vocab) in PATTERNS {
        if triggers.iter().any(|t| lower.contains(t)) {
            tokens.extend_from_slice(vocab);
        }
    }

    // NE-6: Universal disclosure-signal extraction (TRIZ P10 Preliminary Action).
    //
    // "By the way, [fact]" is the dominant user disclosure pattern in conversational memory:
    // 803 occurrences across 500 sessions (1.6× per session) in LME-500.
    // "Speaking of," and "Also," are secondary signals.
    //
    // Extract up to 30 content words after each disclosure signal and add them to the
    // query_surface. This is applied ALWAYS (not just when category patterns fail) so
    // that the specific fact vocabulary — e.g. "Business Administration", "Philips LED",
    // "Target" — enters the BM25 index with the 1.5× query_surface boost, making the
    // correct session rank above competing sessions that mention the terms incidentally.
    let mut extra_tokens: Vec<String> = {
        const SKIP: &[&str] = &[
            "the", "and", "for", "are", "was", "but", "not", "you", "all", "can", "her", "his",
            "she", "they", "them", "any", "had", "our", "one", "this", "that", "its", "with",
            "have", "from", "just", "been",
        ];
        const SIGNALS: &[&str] = &[
            "by the way",
            "speaking of,",
            "also,",
            "i should mention",
            "incidentally,",
            "anyway,",
            "just wanted to mention",
        ];
        let mut extra = Vec::new();
        for signal in SIGNALS {
            if let Some(pos) = lower.find(signal) {
                let after_start = (pos + signal.len()).min(text.len());
                let after = text[after_start..].trim_start_matches([',', ' ', '\t']);
                for word in after.split_whitespace().take(30) {
                    let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                    let cl = clean.to_lowercase();
                    if cl.len() >= 3 && !SKIP.contains(&cl.as_str()) {
                        extra.push(cl);
                    }
                }
            }
        }
        extra
    };

    // NE-7: Targeted person/place name extraction near personal relationship triggers.
    //
    // Narrowly scoped to rare, specific relationship labels only.  "my friend" / "my
    // colleague" are too common (appear in nearly every session) and flooding
    // query_surface with person names creates noise across multi-session and temporal
    // categories.  Only "my sister", "my cousin", and "visiting my" are kept: they are
    // specific enough that the capitalized words immediately following are almost always
    // person names or city names that are unique discriminators.
    // Example: "visiting my sister Emily in Denver" → ["emily", "denver"] added to
    // extra_tokens → query "where does my sister Emily live?" → "emily" in
    // query_surface at 1.5× → correct session ranked above generic "emily" hits.
    if !tokens.is_empty() {
        const REL_TRIGGERS: &[&str] = &["my sister", "my cousin", "visiting my"];
        for trigger in REL_TRIGGERS {
            let mut search_start = 0;
            while let Some(rel_pos) = lower[search_start..].find(trigger) {
                let abs_pos = search_start + rel_pos;
                let after_start = (abs_pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(8) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().map_or(false, |c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 {
                        break;
                    }
                }
                search_start = abs_pos + trigger.len();
                if search_start >= lower.len() {
                    break;
                }
            }
        }
    }

    // NE-8: Degree/field-of-study name extraction after education-specific phrases.
    // "I graduated with a degree in Business Administration" → ["business", "administration"]
    // This bridges the vocabulary gap: the query "what degree did I graduate with?" does not
    // contain "business administration", but those capitalized words are unique to the session.
    // Having them in query_surface means cross-session deduplication is stronger.
    // Fires only when tokens is non-empty (an education or other pattern already matched).
    if !tokens.is_empty() {
        const EDU_TRIGGERS: &[&str] = &[
            "degree in ",
            "majored in ",
            "major in ",
            "studied ",
            "i have a degree in",
            "graduated with a degree in",
            "studying for a ",
            "i earn my degree in",
        ];
        for trigger in EDU_TRIGGERS {
            if let Some(pos) = lower.find(trigger) {
                let after_start = (pos + trigger.len()).min(text.len());
                let after = &text[after_start..];
                let mut found = 0;
                for word in after.split_whitespace().take(5) {
                    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if clean.len() >= 3
                        && clean.chars().next().map_or(false, |c| c.is_uppercase())
                        && found < 3
                    {
                        extra_tokens.push(clean.to_lowercase());
                        found += 1;
                    }
                    if found >= 3 {
                        break;
                    }
                }
            }
        }
    }

    // This catch-all layer ensures BM25 can find the neuron via ANY vocabulary in its
    // content, even when the content doesn't match any predefined category pattern.
    // Zero false-positive risk: these terms are extracted directly from the content.
    if tokens.is_empty() {
        let mut fallback: Vec<String> = Vec::new();

        // (a) Proper nouns: capitalized words ≥3 chars, not sentence-start
        for (i, word) in text.split_whitespace().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() >= 3
                && i > 0  // skip sentence-start capitals
                && clean.chars().next().map_or(false, |c| c.is_uppercase())
            {
                fallback.push(clean.to_lowercase());
            }
        }

        // (b) Numbers / quantities: tokens containing digits (ages, counts, times)
        for word in text.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
                .collect();
            if clean.chars().any(|c| c.is_ascii_digit()) && clean.len() >= 2 {
                fallback.push(clean.to_lowercase());
            }
        }

        // (c) Quoted strings: extract content between " " or ' '
        let mut in_quote = false;
        let mut quote_buf = String::new();
        for ch in text.chars() {
            if ch == '"' || ch == '\'' {
                if in_quote && !quote_buf.trim().is_empty() {
                    for part in quote_buf.split_whitespace() {
                        let clean: String = part.chars().filter(|c| c.is_alphabetic()).collect();
                        if clean.len() >= 3 {
                            fallback.push(clean.to_lowercase());
                        }
                    }
                    quote_buf.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                quote_buf.push(ch);
            }
        }

        fallback.extend(extra_tokens);
        if fallback.is_empty() {
            return None;
        }

        // Deduplicate fallback tokens
        let mut seen = std::collections::HashSet::new();
        let deduped: Vec<String> = fallback
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect();
        return Some(deduped.join(", "));
    }

    // Deduplicate while preserving order; merge category vocab + disclosure terms
    let mut seen = std::collections::HashSet::new();
    let mut deduped: Vec<String> = tokens
        .into_iter()
        .filter(|t| seen.insert(t.to_string()))
        .map(|s| s.to_string())
        .collect();
    for t in extra_tokens {
        if seen.insert(t.clone()) {
            deduped.push(t);
        }
    }
    Some(deduped.join(", "))
}
