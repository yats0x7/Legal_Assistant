import os

if not os.path.exists('data'):
    os.makedirs('data')

# We are using the NEW laws (BNS 2023) because your assistant should be modern.
legal_text = """
THE BHARATIYA NYAYA SANHITA, 2023 (BNS)

CHAPTER 1: PRELIMINARY
Section 1. Short title, commencement and application.
(1) This Act may be called the Bharatiya Nyaya Sanhita, 2023.
(2) It shall come into force on such date as the Central Government may, by notification in the Official Gazette, appoint.

CHAPTER 5: OFFENCES AGAINST WOMAN AND CHILD
Section 63. Rape.
A man is said to commit "rape" if he satisfies any of the following circumstances:
(a) penetrates his penis, to any extent, into the vagina, mouth, urethra or anus of a woman or makes her to do so with him or any other person; or
(b) inserts, to any extent, any object or a part of the body, not being the penis, into the vagina, the urethra or anus of a woman or makes her to do so with him or any other person; or
(c) manipulates any part of the body of a woman so as to cause penetration into the vagina, urethra, anus or any part of body of such woman or makes her to do so with him or any other person; or
(d) applies his mouth to the vagina, anus, urethra of a woman or makes her to do so with him or any other person.
Punishment: Rigorous imprisonment of not less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.

Section 69. Sexual intercourse by employing deceitful means.
Whoever, by deceitful means or making by promise to marry to a woman without any intention of fulfilling the same, and has sexual intercourse with her, such sexual intercourse not amounting to the offence of rape, shall be punished with imprisonment of either description for a term which may extend to ten years and shall also be liable to fine.
Explanation.—"Deceitful means" shall include inducement for, or false promise of employment or promotion, inducement or false identity of marriage.

CHAPTER 6: OFFENCES AFFECTING THE HUMAN BODY
Section 103. Punishment for murder.
(1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.
(2) When a group of five or more persons acting in concert commits murder on the ground of race, caste or community, sex, place of birth, language, personal belief or any other like ground, each member of such group shall be punished with death or with imprisonment for life, and shall also be liable to fine.

Section 304. Snatching.
(1) Theft is "snatching" if, in order to commit theft, the offender suddenly or quickly or forcibly seizes or secures or grabs or takes away from any person or from his possession any movable property.
(2) Whoever commits snatching shall be punished with imprisonment of either description for a term which may extend to three years, and shall also be liable to fine.
"""

with open('data/bns_sample.txt', 'w', encoding='utf-8') as f:
    f.write(legal_text)

print("✅ Legal text file created at: data/bns_sample.txt")