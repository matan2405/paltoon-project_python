**Name of Research Project:** Game-Theoretic Control for Autonomous Vehicle Merging

**Research Protocol:**

**Goals**:

This study uses a PC-based driving simulator (Unity) to evaluate a Nash equilibrium-based shared control algorithm for highway platoon merging. Participants will drive a simulated ego vehicle equipped with a Logitech G29 steering wheel and pedals while the system aids through a real-time Iterative Best Response solver. The study compares three conditions:

1.  **Manual control:** the driver is the sole source of input; the Nash solver is disabled.
2.  **Nash-Shared (50% authority):** the driver and the autonomous system share control with equal weighting (λ = 1.0). The system intervenes smoothly in both longitudinal (speed/gap) and lateral (lane-change) axes based on a real-time safety field.
3.  **Autonomous (100% authority):** the system overrides driver input entirely; the participant may remain at the wheel, but the vehicle is controlled autonomously.

The comparison is based on objective performance data (vehicle dynamics: acceleration, braking, inter-vehicle gap, Time-to-collision (TTC)) and subjective questionnaires (trust, mental workload, system acceptance).

The experiment will be conducted in the Mechanical Engineering Department at Dr. Shai Arogeti’s lab, Ben-Gurion University of the Negev.

**Participants:**

The study will be carried out on approximately 12–15 participants (12 for analysis; 3 reserve). Inclusion criteria:

-   BGU students aged 21–35
-   Valid Category B driving license with at least two years of driving experience
-   Normal or corrected-to-normal visual acuity
-   No self-reported history of simulator sickness

Participants will be recruited on a voluntary basis by word of mouth among colleagues and students, and through the Human Factors course taught by Dr. Avinoam Borowsky. The study session lasts approximately 30 minutes. Participants enrolled in the Human Factors course will receive one bonus point in the course grade for participation; no other compensation is provided.

**Tools:**

PC-based driving simulator. The simulator is in the Mechanical Engineering Department, Building 57, BGU. It comprises a standard PC with a GPU, a 27-inch LCD monitor at 70 cm viewing distance, a Logitech G29 driving wheel with throttle and brake pedals, and a stable driver’s seat.

The driving simulation software is Unity 6000.3.22f1 with the Universal Render Pipeline. The virtual highway environment includes a four-vehicle autonomous platoon and a scripted disturbance scheduler that triggers two types of pre-programmed traffic events during the session. An on-screen HUD shows vehicle speed, current driving phase, and (in the autonomous condition) an “AUTO” indicator.

**Experimental Design:**

Each participant experiences all three control conditions (manual, shared, autonomous) in a single session. The experiment uses a within-subjects design, meaning each participant experiences all three authority levels (0%, 50%, 100%). To prevent learning and carry-over effects from distorting the results, the order in which participants encounter these conditions is counterbalanced across six possible orderings (a Latin square). Each order is assigned to exactly 2 participants.

The session consists of two parts:

Part 1 — Merging task (2 runs, counterbalanced). The participant performs a join-middle platoon merge twice: once in manual mode and once in Nash-Shared mode. The order is counterbalanced across participants.

Part 2 — Following task (3 blocks, counterbalanced). After a successful merge, the participant drives continuously inside the platoon across three consecutive blocks, each under a different authority level (0%, 50%, 100%). Each block lasts 60 seconds and contains two scripted traffic disturbances. A 20-second reset period separates blocks. Block order is counterbalanced via Latin square.

Two scripted disturbances occur once each per block, in a fixed sequence within the block:

**(a) Lead vehicle acceleration.** The platoon's lead vehicle smoothly accelerates by Δv = +3 m/s (approximately 10.8 km/h) over two seconds, then holds the new speed. This opens the gap between the participant's vehicle and the vehicle directly ahead, creating a forward gap-error that requires either increased throttle (manual condition) or triggers the shared/autonomous controller to close the gap. The disturbance lasts until a new steady-state following gap is re-established (typically within 10–15 seconds).

**(b) External vehicle entering the platoon behind the participant.** A scripted vehicle approaches from the rear and enters the gap between the participant's vehicle and the vehicle immediately behind, with an initial time-to-collision (TTC) of approximately 3 seconds at the moment of entry. This generates a rear-proximity hazard that activates the rear-facing component of the shared control authority field. The intruding vehicle then matches platoon speed, and the disturbance is resolved when inter-vehicle spacing stabilizes.

**Procedure:**

1.  The experimenter (Matan Sason) demonstrates one manual and one Nash-Shared merge before the participant’s session begins (not included in analysis).
2.  The participant reads and signs the Informed Consent Form.
3.  The participant completes a brief demographics and driving history questionnaire.
4.  The participant performs a 2-minute free-drive familiarization run on an empty road to get comfortable with the G29 setup.
5.  **Run A** (merging task, condition per counterbalancing order): the participant merges into the platoon. On completion, the participant fills in the AV-TLX workload questionnaire.
6.  **Run B** (merging task, the other condition): the participant merges again. On completion, the participant fills in the AV-TLX questionnaire.
7.  **Block α, β, γ** (following task, order per counterbalancing): the participant drives inside the platoon under each authority level. After each block, the participant fills in the AV-TLX questionnaire.
8.  At the end of the session: TiA-19 Trust questionnaire, HMII interaction questionnaire, Van der Laan Acceptance questionnaire.
9.  Oral debriefs: the experimenter explains the study goals, answers questions, and thanks the participant.

Total session duration: approximately **30 minutes**.

**Data Security & Disclosure:**

-   All data collected will be saved on a password-protected computer in Dr. Arogeti’s lab (access restricted to the research team).
-   Each participant is assigned a randomized subject ID dissociated from the consent form. This number is used for all simulator CSV files and questionnaire responses.
-   Consent forms are stored in a locked cabinet in the lab. Only the research team can link identifying details to performance data.
-   No identifying details will appear in any publication. All future publications will report aggregated, anonymized data only.

**References:**

1.  Flad, M., Frohlich, L., & Hohmann, S. (2017). Cooperative shared control driver assistance systems based on motion primitives and differential games. *IEEE Transactions on Human-Machine Systems*, 47(5), 711–722.
2.  Na, X., & Cole, D. J. (2022). Theoretical and experimental study of a game-theoretic controller for vehicle overtaking. *IEEE Transactions on Human-Machine Systems*, 52(3), 411–421.
3.  Lazcano, R., et al. (2021). Driver preferences in shared control for lane keeping. *IEEE Access*, 9, 137396–137411.
4.  Marcano, M., et al. (2020). A review of shared control for automated vehicles. *IEEE Transactions on Human-Machine Systems*, 50(6), 547–557.

**Appendices:**

  
**I.** **Instructions**

A printed copy of the Instructions will be provided for each participant when they arrive.

אנא אשר את טופס הסכמה לניסוי

1.  אנא מלא את השאלון הקצר.
2.  ‏בדקות הקרובות תתבקש לנהוג בסימולטור כדי להרגיש בנוח בנהיגה בסימולטור. התפקיד שלך הוא לנהוג בסימולטור (בשימוש בגז ובבלם) כאילו זה העולם האמיתי. אנא נהג כמה שאתה צריך עד שאתה מרגיש בנוח עם הסביבה ועם רמת הנהיגה שלך.
3.  ‏נסה לשמור על מרחק אחיד מהרכב שלפניך כסימן שאתה שולט בתאוצה ובמהירות הרכב. כשאתה מרגיש בנוח עם הסימולטור, אנא יידע את הנסיין והוא יאתחל את הסימולטור עם שיירת הרכבים.
4.  ‏לאחר שהסימולציה תתחיל, תתבקש לבצע מיזוג לתוך השיירה שהוא מעבר לנתיב הימני ולהתייצב בין רכב 2 לרכב 3. שים לב:

-   אם המערכת במצב **שליטה משותפת**: התנהג כאילו השליטה בידיים שלך באופן מידי.
-   אם המערכת במצב **אוטונומי מלא**: המתן להודעה על המסך לפני שאתה מתערב.

1.  ‏לאחר המיזוג תנהג בתוך השיירה לאורך מספר דקות תחת שלוש רמות סיוע שונות (ידני / משותף / אוטונומי). המערכת תעדכן אותך לגבי רמת הסיוע הפעילה דרך הצג. במהלך הנסיעה יתרחשו מספר אירועי תנועה — אנא הגב אליהם כפי שהיית מגיב בעולם האמיתי.
2.  ‏הנסיין יודיע לך כשאתה צריך להפסיק את הסימולטור. לאחר כל שלב תתבקש למלא שאלון קצר, ובסוף הניסוי שאלון סיכום.
3.  ‏אם בכל עת אתה חש אי-נוחות — אנא הודע לנסיין מיידית. ניתן להפסיק את הניסוי בכל שלב ללא כל השלכות.

**II**. **Demographics Questionnaire to complete BEFORE the session**

Since all the participants are expected to speak Hebrew, the questionnaires will be in Hebrew.

-   **Demographics Questionnaire (Appendix IX)** - collects participant background information: age, gender, years of driving experience, annual mileage, and prior experience with autonomous vehicles.

**III. Questionnaires to complete DURING the session**

-   **AV-TLX (Appendix VI)** - measures the participant's subjective mental workload while driving an autonomous vehicle. Administered 5 times: after each of the 2 merging runs and after each of the 3 FOLLOWING blocks, to track workload changes across the session.

**IV**. **Questionnaire to complete AFTER the session**

-   **TiA-19 (Appendix V):** measures the participant's trust in the autonomous system across 5 dimensions: reliability, benevolence, understandability, propensity to trust, and overall trust.
-   **HMII (Appendix VII):** measures the quality of the driver-system interaction, including perceived controllability, comfort, and acceptance.
-   **Van der Laan Acceptance Scale (Appendix VIII):** measures system acceptance across two dimensions: usefulness and satisfaction.

**V**. **TiA-19**

**אוניברסיטת בן-גוריון בנגב | הפקולטה למדעי ההנדסה | המחלקה להנדסת מכונות**

**נספח ה: שאלון אמון במערכות אוטומטיות (TiA-19)**

מחקר: שליטה משותפת מבוססת שיווי משקל של נאש למיזוג רכב אוטונומי בפלטון

הוראות למשתתף:

לפניך 19 היגדים על המערכת שבה נהגת. עבור כל היגד, סמן ✕ במשבצת המתאימה שמייצגת את מידת ההסכמה שלך עם ההיגד. הסקאלה נעה מ"מסכים בכלל לא" (1) ועד "מסכים לחלוטין" (5). אין תשובות נכונות או שגויות — אנו מעוניינים בדעתך האישית בלבד.

| מזהה משתתף: _______________ | מצב: _______________ | תאריך: ___/___/______ |
| --- | --- | --- |

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) | אין   תשובה |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | המערכת יכולה לפרש מצבים באופן נכון. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | מצב המערכת היה תמיד ברור לי. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | אני מכיר כבר מערכות דומות. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | המפתחים ראויים לאמון. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | צריך להיזהר עם מערכות אוטומטיות לא מוכרות. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | המערכת פועלת באופן אמין. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | המערכת מגיבה באופן בלתי צפוי. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | המפתחים לוקחים ברצינות את שלומי. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | אני סומך על המערכת. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | תקלה במערכת היא סבירה. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 11 | הצלחתי להבין מדוע דברים קורים. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 12 | אני נוטה יותר לסמוך על מערכת מאשר לא לסמוך עליה. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 13 | המערכת מסוגלת לקחת על עצמה משימות מורכבות. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 14 | אני יכול להסתמך על המערכת. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 15 | המערכת עלולה לעשות טעויות לפעמים. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 16 | קשה לזהות מה המערכת תעשה בהמשך. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 17 | כבר השתמשתי במערכות דומות. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 18 | מערכות אוטומטיות בדרך כלל פועלות היטב. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 19 | אני בטוח ביכולות של המערכת. | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

ניקוד (למחקר בלבד — לא למילוי על ידי המשתתף):

השאלון מבוסס על Körber (2018) ומודד 6 ממדים:

• מהימנות/יכולת (Reliability/Competence): פריטים 1, 6, 7\*, 10\*, 15\*, 18

• מובנות/צפיוּת (Understandability/Predictability): פריטים 2, 11, 16\*

• נטייה לסמוך על אוטומציה (Propensity to Trust): פריטים 5\*, 12

• כוונות המפתחים (Intention of Developers): פריטים 4, 8

• היכרות (Familiarity): פריטים 3, 17

• אמון במערכת (Trust in Automation): פריטים 9, 13, 14, 19

\* פריטים מהופכים (Reversed): יש להפוך את הסקאלה לפני חישוב (6 - x).

Körber, M. (2018). Theoretical Considerations and Development of a Questionnaire to Measure Trust in Automation.

**VI. AV-TLX**

**אוניברסיטת בן-גוריון בנגב | הפקולטה למדעי ההנדסה | המחלקה להנדסת מכונות**

**נספח ו׳: שאלון AV-TLX (עומס מנטלי בנהיגה אוטונומית)**

מחקר: שליטה משותפת מבוססת שיווי משקל של נאש למיזוג רכב אוטונומי בפלטון

הוראות למשתתף:

לפניך 19 היגדים המתייחסים לחוויית העומס המנטלי שלך במהלך הסבב שזה עתה סיימת. עבור כל היגד, סמן ✕ במשבצת המתאימה שמייצגת את מידת ההסכמה שלך. הסקאלה נעה מ-1 (מסכים בכלל לא) עד 7 (מסכים לחלוטין). אין תשובות נכונות או שגויות — דווח בכנות על החוויה שלך.

| מזהה משתתף: _______________ | מצב: _______________ | תאריך: ___/___/______ |
| --- | --- | --- |

דרישה פיזית (Physical Demand)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | החזרת השליטה הידנית מהנהיגה האוטונומית גרמה לי אי-נוחות פיזית. | TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | המרחב המצומצם לפעילויות אישיות במהלך הנהיגה האוטונומית יצר לחץ פיזי על גופי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

תסכול (Frustration)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | העובדה שהיה לי תפקיד פחות משמעותי בהשוואה ליכולות המתקדמות של הרכב יצרה לי עומס מנטלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | מספר הבקשות של הרכב לקחת שליטה, במיוחד במצבים פשוטים, הפעילו את מחשבתי באופן מתמיד. | TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

ביצוע אישי (Personal Performance)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5R | בזמן שהרכב נסע באופן אוטונומי, יכולתי להתרכז בפעילויות האישיות שלי ללא מאמץ מנטלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6R | לקחתי בחזרה שליטה על הרכב כנדרש, מבלי לחוש מאמץ מנטלי. | TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

לחץ הקשרי (Contextual Stress)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | תנאים מכבידים כמו מזג אוויר גרוע, תאורה חלשה או כביש מורכב הגבירו את העומס המנטלי שלי הן במהלך הנהיגה האוטונומית והן במהלך זמן ה-Take-Over. | AD/TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8R | במסעות ארוכים, השימוש במצב האוטונומי הפחית את העומס המנטלי שלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9R | העומס המנטלי שלי פחת כשסמכתי על החיישנים של הרכב שיזהו את הסביבה. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

דרישה זמנית (Temporal Demand)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10R | זמן מספיק לביצוע ה-Take-Over הפחית את העומס המנטלי שלי. | TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

דרישה חזותית/קולית/מגעית (Visual/Vocal/Tactile Demand)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | גודל התצוגה, מיקומה ועיצובה הגבירו את העומס המנטלי בשימוש בהם. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 12 | הבהירות וכמות המידע בתצוגות היו תובעניות מנטלית. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 13 | האינטראקציות הקוליות של הרכב מעוצבות היטב וגורמות למסע להיות פשוט מנטלית. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 14R | אינטראקציות מגעיות אפקטיביות עם הרכב הפחיתו את העומס המנטלי שלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

דרישה תפיסתית ברמה גבוהה (High-Level Perceptual Demand)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15R | השקיפות של הרכב בקבלת החלטות הפחיתה את העומס המנטלי שלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 16R | דפוס הנהיגה של הרכב, שהתאים להעדפותיי, הפחית את העומס המנטלי שלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

חששות בטיחות (Safety Concerns)

| # | היגד | שימוש | מסכים   בכלל לא   (1) | (2) | (3) | ניטרלי   (4) | (5) | (6) | מסכים   לחלוטין   (7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | ביצוע משימות שאינן קשורות לנהיגה במרחב המצומצם מאחורי ההגה יצר לי עומס מנטלי. | AD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 18R | כפתור עצירת החירום נתן לי תחושה של שליטה וביטחון והפחית את העומס המנטלי שלי. | TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 19 | בשל המודעות המצבית הנמוכה שלי, בקשת ה-Take-Over גרמה לעומס מנטלי משמעותי. | TO | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

הערות על השימוש בשאלון בניסוי הנוכחי:

• פריטים המסומנים AD (Automated Driving) או AD/TO — רלוונטיים לניסוי (רקע לבן).

• פריטים המסומנים TO (Take-Over) — פריטים 1, 4, 6, 10, 18, 19 (רקע אפור).

ניקוד (למחקר בלבד):

סקאלת Likert 7 נקודות. פריטים מסומנים ב-R (Reversed) — יש להפוך לפני חישוב (8 - x).

פריטים מהופכים: 5, 6, 8, 9, 10, 14, 15, 16, 18.

שני אופני ניקוד אפשריים:

1\. ציון כללי: ממוצע של כל 19 הפריטים (טווח 1-7). ציון גבוה יותר = עומס גבוה יותר.

2\. פירוט לפי 8 תת-סולמות: Physical, Frustration, Personal Performance, Contextual Stress, Temporal, Visual/Vocal/Tactile, High-Level Perceptual, Safety Concerns.

3\. חלוקה לשני סוגי משימות: AD (13 פריטים) ו-TO (7 פריטים, אחד משותף).

מהימנות מדווחת במאמר המקורי: Cronbach's alpha = 0.86.

תוקף תוכן: CVR > 0.78.

התאמת השאלון לניסוי הנוכחי (Take-Over items):

בעיצוב הניסוי הנוכחי אין בקשת השתלטות פתאומית של המערכת (Take-Over Request, TOR). על כן, פריטי ה-TO של השאלון סווגו כדלקמן ביחס לניתוח הסטטיסטי:

• פריטים רלוונטיים לניתוח (2 פריטים): 1, 6R — הפריטים הללו מתייחסים לחוויית לקיחת השליטה מהמערכת האוטונומית, שרלוונטית למעבר מהמיזוג האוטונומי לבלוקים במצב ידני (0%) או משותף (50%).

• פריטים שיסומנו כ"לא בניתוח" (4 פריטים): 4, 10R, 18R, 19 — פריטים אלו מתייחסים במפורש לבקשות TOR, זמן תגובה ל-TOR, וכפתור עצירת חירום, שאינם קיימים בעיצוב הניסוי.

סה"כ פריטים בניתוח: 13 פריטי AD + 2 פריטי TO רלוונטיים = 15 פריטים.

Mosaferchi, S., Mortezapour, A., Liebherr, M., Villecco, F., & Naddeo, A. (2025). AV-TLX for Measuring (Mental) Workload While Driving AVs. HCII 2025, LNCS 15817.

**VII**. **HMII**

**אוניברסיטת בן-גוריון בנגב | הפקולטה למדעי ההנדסה | המחלקה להנדסת מכונות**

**נספח ז׳: שאלון HMII (אינטראקציה נהג-מערכת)**

מחקר: שליטה משותפת מבוססת שיווי משקל של נאש למיזוג רכב אוטונומי בפלטון

הוראות למשתתף:

לפניך 33 היגדים המתייחסים לאינטראקציה שלך עם המערכת האוטונומית ברכב. עבור כל היגד, סמן את מידת ההסכמה שלך על סקאלה מ-1 (מסכים בכלל לא) עד 5 (מסכים לחלוטין). התייחס לחוויה שלך במהלך הסבבים שזה עתה סיימת.

| מזהה משתתף: _______________ | מצב: _______________ | תאריך: ___/___/______ |
| --- | --- | --- |

קונפליקט (Conflict)

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | דחיתי את הפעולה שהעדיפה המערכת. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | שנינו יכולנו להשיג את התוצאות שהעדפנו במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | התוצאות שהעדפנו במצב הזה היו בקונפליקט. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | המערכת העדיפה תוצאה אחרת ממני במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | העדפתי תוצאה אחרת מהמערכת במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |

תלות עתידית: מערכת כלפי אדם (Future Interdependence S→H)

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | תוצאת המצב הזה משפיעה על אופן האינטראקציה של המערכת אתי בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | התנהגותי במצב הזה משפיעה על אופן האינטראקציה של המערכת אתי בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | התנהגותי במצב הזה משפיעה על אופן ההתנהגות של המערכת בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | להתנהגותי במצב הזה אין השפעה על אופן התנהגות המערכת בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |

תלות עתידית: אדם כלפי מערכת (Future Interdependence H→S)

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | תוצאת המצב הזה משפיעה על אופן האינטראקציה שלי עם המערכת בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | התנהגות המערכת במצב הזה משפיעה על אופן האינטראקציה שלי איתה בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | להתנהגות המערכת במצב הזה יש השפעה על אופן התנהגותי בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | להתנהגות המערכת במצב הזה אין השפעה על אופן התנהגותי בעתיד. | ☐ | ☐ | ☐ | ☐ | ☐ |

ודאות מידע: מערכת כלפי אדם (Information Certainty S→H)

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | המערכת מבינה כיצד פעולותיה משפיעות עליי. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | המערכת יודעת מה אני מתכנן לעשות במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | המערכת מודעת לפעולה המתוכננת שלי במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | המערכת יודעת מדוע אני מעדיף פעולה מסוימת. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | המערכת אינה יודעת מה אני מתכנן לעשות במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |

ודאות מידע: אדם כלפי מערכת (Information Certainty H→S)

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | אני מבין כיצד פעולתי משפיעה על המערכת. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | אני יודע מה המערכת מתכננת במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | אני מיודע לגבי הפעולה המתוכננת של המערכת במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | אני יודע מדוע המערכת מעדיפה פעולה מסוימת. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | אינני יודע מה המערכת מתכננת במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |

תלות הדדית (Mutual Dependence)

| # | היגד | מסכים   בכלל לא   (1) | לא מסכים   כל כך   (2) | לא מסכים   ולא לא מסכים   (3) | מסכים   בחלקו   (4) | מסכים   לחלוטין   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | אנחנו תלויים זה בזה במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | שנינו תלויים זה בזה במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | אנחנו זקוקים זה לזה כדי להשיג את התוצאה הטובה ביותר במצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | התוצאה של כל אחד מאיתנו תלויה בהתנהגות של האחר. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | אנחנו זקוקים זה לזה כדי לפתור את המצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | אנחנו צריכים לעבוד יחד כדי להתמודד עם המצב הזה. | ☐ | ☐ | ☐ | ☐ | ☐ |

כוח (Power) — סקאלה שונה

לפריטים הבאים, הסקאלה נעה בין "בהחלט המערכת" (1) לבין "בהחלט אני" (5).

| # | היגד | בהחלט   המערכת   (1) | יותר   המערכת   (2) | ניטרלי   (3) | יותר   אני   (4) | בהחלט   אני   (5) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | מי הרגשת שהייתה לו ההשפעה הגדולה ביותר על מה שקרה במצב הזה? | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | מי הרגשת שהייתה לו ההשפעה הגדולה ביותר על הפעולה שבוצעה? | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | מי הרגשת שהייתה לו ההשפעה הקטנה ביותר על מה שקרה במצב הזה? | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | מי לדעתך הייתה לו ההשפעה הקטנה ביותר על הפעולה שבוצעה? | ☐ | ☐ | ☐ | ☐ | ☐ |

ניקוד (למחקר בלבד):

• פריטים מהופכים (יש להפוך לפני חישוב הממוצע):

\- Conflict: פריט 2

\- Future Interdependence S→H: פריט 4

\- Future Interdependence H→S: פריט 4

\- Information Certainty S→H: פריט 5

\- Information Certainty H→S: פריט 5

\- Power: פריטים 3 ו-4

• ציון כל מימד = ממוצע של כל הפריטים במימד (לאחר היפוך).

Woide, M., Stiegemeier, D., Pfattheicher, S., & Baumann, M. (2021). Transportation Research Part F, 83, 424-439

**VIII. Van der Laan Acceptance Scale**

**אוניברסיטת בן-גוריון בנגב | הפקולטה למדעי ההנדסה | המחלקה להנדסת מכונות**

**נספח ט׳: שאלון קבלת מערכות (Van der Laan Acceptance Scale)**

מחקר: שליטה משותפת מבוססת שיווי משקל של נאש למיזוג רכב אוטונומי בפלטון

הוראות למשתתף:

לפניך 9 שורות, כל אחת מציגה שני תיאורים הפוכים של המערכת בה נהגת. בכל שורה, סמן ✕ באחת מ-5 המשבצות שבין שני התיאורים, בהתאם למידה שבה התיאור מתאים לחוויה שלך. משבצת שמאלית = התיאור השמאלי מתאים חזק, משבצת ימנית = התיאור הימני מתאים חזק, משבצת אמצעית = ניטרלי.

| מזהה משתתף: _______________ | מצב: _______________ | תאריך: ___/___/______ |
| --- | --- | --- |

| # | תיאור ימני | +2 | +1 | 0 | −1 | −2 | תיאור שמאלי |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | שימושי   (Useful) | ☐ | ☐ | ☐ | ☐ | ☐ | חסר תועלת   (Useless) |
| 2 | נעים   (Pleasant) | ☐ | ☐ | ☐ | ☐ | ☐ | לא נעים   (Unpleasant) |
| 3 ⚠ | רע   (Bad) | ☐ | ☐ | ☐ | ☐ | ☐ | טוב   (Good) |
| 4 | יפה   (Nice) | ☐ | ☐ | ☐ | ☐ | ☐ | מעצבן   (Annoying) |
| 5 | יעיל   (Effective) | ☐ | ☐ | ☐ | ☐ | ☐ | מיותר   (Superfluous) |
| 6 ⚠ | מרגיז   (Irritating) | ☐ | ☐ | ☐ | ☐ | ☐ | חביב   (Likeable) |
| 7 | מסייע   (Assisting) | ☐ | ☐ | ☐ | ☐ | ☐ | חסר ערך   (Worthless) |
| 8 ⚠ | לא רצוי   (Undesirable) | ☐ | ☐ | ☐ | ☐ | ☐ | רצוי   (Desirable) |
| 9 | מעורר ערנות   (Alerting) | ☐ | ☐ | ☐ | ☐ | ☐ | מרדים   (Sleep-inducing) |

חישוב ניקוד (למחקר בלבד):

הסקאלה מודדת שני מימדים:

• שימושיות (Usefulness) = ממוצע של פריטים 1, 3, 5, 7, 9 (טווח: −2 עד +2)

• שביעות רצון (Satisfying) = ממוצע של פריטים 2, 4, 6, 8 (טווח: −2 עד +2)

⚠ פריטים מהופכים (יש להפוך לפני חישוב): 3, 6, 8

ציון גבוה יותר = קבלה טובה יותר של המערכת.

Van der Laan, J. D., Heino, A., & de Waard, D. (1997). Transportation Research Part C, 5(1), 1-10

**IX. Demographics Questionnaire**

**אוניברסיטת בן-גוריון בנגב | הפקולטה למדעי ההנדסה | המחלקה להנדסת מכונות**

**נספח י׳: שאלון דמוגרפי**

מחקר: שליטה משותפת מבוססת שיווי משקל של נאש למיזוג רכב אוטונומי בפלטון

הוראות למשתתף:

אנא מלא את הפרטים הבאים לפני תחילת הניסוי. השאלון אנונימי — הפרטים ישמשו לצורכי ניתוח סטטיסטי בלבד ולא יקושרו לזהותך האישית.

| מספר משתתף: \_\_\_\_\_\_\_\_\_\_\_\_ | תאריך: \_\_\_/\_\_\_/\_\_\_\_ |
| --- | --- |

1\. מה גילך?

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

2\. מה מינך?

○ זכר     ○ נקבה     ○ אחר: \_\_\_\_\_\_\_\_\_\_\_\_

3\. כמה שנות נהיגה יש לך?

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

4\. האם יש ברשותך רכב? אם כן, פרט כמה קילומטרים (בערך) אתה עושה בשנה?

○ לא

○ כן — כ-\_\_\_\_\_\_\_\_\_\_\_ ק”מ בשנה

5\. האם נהגת ברכב עם יכולות אוטונומיות (לדוגמה: כביש מהיר אוטונומי, בלימה חירום אוטומטית, שמירת נתיב)? אם כן, פרט מה.

○ לא

○ כן — \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

6\. האם שיחקת בעבר במשחקי מחשב או סימולטורים? (כולל משחקי נהיגה)

○ לא

○ לעיתים רחוקות

○ לעיתים קרובות

**X. Consent Form**

(To be completed by participants on arrival)

Consent Form

Principal Investigators:

Shai Arogeti,

PhD Mechanical Engineering Dept.

Ben Gurion University of the Negev.

Avinoam Borowsky

PhD Industrial Engineering and Management Dept.

Ben Gurion University of the Negev

You are invited to participate in a study conducted by Ben-Gurion University of the Negev.

The purpose of this study is to evaluate a shared control algorithm designed to assist drivers during autonomous vehicle platoon merging on a highway. Your participation will help us understand how drivers respond to different levels of automated assistance and improve the safety and comfort of shared human-machine control systems.

If you agree to be part of the study, you will drive a PC-based driving simulator (screen-based, no motion platform) in Dr. Arogeti’s lab, Mechanical Engineering Building 57, BGU. You will use a Logitech G29 steering wheel and pedals. During the session, the system will switch between three modes: fully manual, shared (human + system), and fully autonomous.

The session takes approximately **30 minutes**, during which you will also be asked to complete short questionnaires.

Signing this form means that you consent to participate in the procedures described above.

**NOTE:**

-   **We will not collect any identifying details about you.**
-   **The only data we collect includes data from the driving simulator and answers to the questionnaire and this data cannot be linked to any identifying details about you.**
-   **No person outside the research team will have any access to your information.**
-   **Your identity will not be revealed in any publication resulting from this study.**

Please also note that your participation in this study is **voluntary**. Even if you sign this form, you may choose to stop driving and/or not answer any questionnaire at any time, and you may withdraw your consent to participate at any time. Should you decide not to participate or to withdraw from the study, this will have no consequences for you whatsoever.

Participants who are enrolled in the Human Factors course taught by Dr. Avinoam Borowsky will receive one bonus point in the course grade for participation in this study. There is no other form of compensation.

If at any point you feel uncomfortable, please inform the experimenter immediately. The session can be stopped at any time.

If you have any questions or concerns about this study, please contact: **Dr. Shai Arogeti** at arogeti@bgu.ac.il

I understand the information presented in this document, and I had the opportunity to ask questions and receive answers from the investigator.

**NOTE: This Consent form is the only form that includes your name (NOT your ID number).**

**To ensure additional privacy you will be assigned a randomized subject ID that will be used in the collection of data.**

I AGREE TO VOLUNTARILY PARTICIPATE IN THE STUDY:

Name: Participant Signature Date

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Name: Experimenter Signature Date

Matan Sason \_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_

I understand the information presented in this document, and I had the opportunity to ask questions and receive answers from the investigator.

BY SIGNING THIS FORM, I VOLUNTARILY AGREE TO PARTICIPATE IN

THE STUDY

Name of Participant Signature Date

\_\_\_\_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_

Name of Experimenter Signature Date

Matan Sason \_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_

טופס הסכמה

(נכתב בלשון זכר אך מופנה לזכר ולנקבה כאחד)

חוקר ראשי:

ד”ר שי ארוגטי

המחלקה להנדסת מכונות

אוניברסיטת בן-גוריון בנגב

ד"ר אבינועם בורובסקי

‏המחלקה להנדסת תעשייה וניהול

‏אוניברסיטת בן-גוריון בנגב

הנך מוזמן/ת להשתתף במחקר שמבצעת אוניברסיטת בן-גוריון בנגב.

מטרת המחקר היא לבחון אלגוריתם שליטה משותפת (Shared Control) שנועד לסייע לנהגים בשלב מיזוג רכב לתוך שיירת רכבים אוטונומית בכביש מהיר. השתתפותך תעזור לנו להבין כיצד נהגים מגיבים לרמות סיוע שונות של מערכת אוטומטית, ולשפר את הבטיחות ואת הנוחות של מערכות שליטה משותפות.

אם תסכים/י להשתתף, תנהג/י בסימולטור נהיגה מבוסס מחשב (מסך בלבד, ללא פלטפורמת תנועה) במעבדה של ד”ר ארוגטי, בניין הנדסת מכונות 57, אוניברסיטת בן-גוריון. תשתמש/י בהגה Logitech G29 ובדוושות גז ובלם. במהלך הניסוי המערכת תעבור בין שלושה מצבים: נהיגה ידנית מלאה, שליטה משותפת (נהג + מערכת), ונהיגה אוטונומית מלאה.

מהלך הניסוי אורך כ-30 דקות, במהלכן תתבקש/י גם למלא שאלונים קצרים.

חתימה על טופס זה פירושה הסכמתך להשתתף בהליכים המתוארים לעיל.

הערות חשובות:

• לא נאסוף עליך פרטים מזהים.

• הנתונים היחידים שאנו אוספים הם נתוני דינמיקת הרכב מהסימולטור ותשובותיך לשאלונים — נתונים אלו אינם ניתנים לקישור לפרטים מזהים.

• לאף גורם מחוץ לצוות המחקר לא תהיה גישה למידע שלך.

• זהותך לא תיחשף בכל פרסום הנובע ממחקר זה.

כמו כן, ההשתתפות במחקר זה היא וולונטרית (התנדבותית). גם אם חתמת על טופס זה, תוכל/י לבחור להפסיק לנהוג ו/או לא לענות על שאלון כלשהו בכל עת, ולחזור בך מהסכמתך להשתתף בכל שלב. לא יהיו לכך השלכות עבורך.

משתתפים הרשומים לקורס גורמי אנוש בהנחיית ד"ר אבינועם בורובסקי יקבלו נקודת בונוס אחת בציון הקורס עבור השתתפות במחקר זה. אין פיצוי כספי או פיצוי אחר עבור ההשתתפות.

אם בכל עת תרגיש/י אי-נוחות, אנא הודיע/י לנסיין מיד. ניתן להפסיק את הניסוי בכל שלב.

לשאלות: ד”ר שי ארוגטי — arogeti@bgu.ac.il

אני מבין/ה את המידע המוצג במסמך זה, והייתה לי הזדמנות לשאול שאלות ולקבל תשובות מהחוקר.

הערה: טופס הסכמה זה הוא המסמך היחיד הכולל את שמך. הוא אינו כולל את מספר תעודת הזהות שלך.

כדי להבטיח פרטיות מרבית, יוקצה לך מספר נבחן אנונימי שישמש לכל איסוף הנתונים.

אני מסכים/ה להשתתף בהתנדבות במחקר:

שם: חתימת המשתתף/ת תאריך

\_\_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_

שם: חתימת הנסיין תאריך

מתן ששון \_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_

אני מבין/ה את המידע המוצג במסמך זה, והייתה לי הזדמנות לשאול שאלות ולקבל תשובות מהחוקר.

‏**על ידי חתימה על טופס זה, אני מסכים/ה באופן וולונטרי (התנדבותי) להשתתף במחקר**

‏שם: חתימת המשתתף/ת תאריך

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_

‏שם: חתימת הנסיין תאריך

‏מתן ששון \_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_