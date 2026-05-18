// Default prompt set seeded on fresh installs. db.ts inserts these only
// when prompt_sets is empty, so users who delete or rename the default
// won't see it come back on the next dashboard restart.
//
// The 12-prompt set covers a deliberate cross of framing (close-up /
// medium / wide) × pose (front / three-quarter / profile) × lighting
// (soft / golden hour / studio / overcast / harsh / tungsten / dusk /
// silhouette / bounce), giving a FLUX-2 LoRA evaluator a representative
// "does this adapter generalise?" canvas without needing the user to
// hand-author one.

export interface SeedPromptSet {
    name: string;
    prompts: string[];
}

export const DEFAULT_PROMPT_SETS: SeedPromptSet[] = [
    {
        name: 'diverse-framing-pose-12-klein',
        prompts: [
            '[3:4] A close-up portrait of [trigger] facing the camera squarely, expression quiet, attention unwavering. Diffused morning light spills through a tall window just off-camera left, wrapping softly around the face and catching faint highlights in the iris while shadows fall short and gentle along the jaw. Style: editorial portrait photography. Mood: contemplative.',
            '[3:4] A close-up of [trigger] turned three-quarters toward the lens, gaze drifting just past the camera. The last warm rays of the setting sun graze the rear edge of the face, tracing a thin amber line along the jaw and the curve of the ear, while the front of the face falls into soft shadow lit only by the cool ambient sky. Style: cinematic golden-hour portraiture. Mood: wistful.',
            '[3:4] A close-up profile of [trigger], head turned a clean ninety degrees from the lens, silhouette held against a deep charcoal seamless paper. A single large softbox high and slightly forward sculpts one cheekbone and plants a single sharp catchlight in the eye; the rest of the face slips into controlled shadow. Style: high-end studio portrait. Mood: composed, sculptural.',
            '[3:4] A close-up of [trigger] caught mid-laugh, chin angled upward, eyes almost shut with genuine joy. An overcast sky acts as one enormous diffuser, lighting the face evenly with no hard shadow at all, the faint warmth of skin reading against the cool grey above. Style: candid documentary. Mood: spontaneous, bright.',
            '[3:4] A close-up of [trigger] looking straight into the lens, sunglasses pushed up into the hair. The midday sun blasts down from directly overhead, carving deep shadows beneath the brow and chin and bleaching highlights on the forehead and bridge of the nose, while squint lines crease softly at the corners of the eyes. Style: gritty street photography. Mood: unguarded, warm.',
            '[3:4] A medium portrait of [trigger] seated at a worn wooden café table, hands curled around a ceramic mug, body open to the camera with a small relaxed smile. Tungsten pendant lamps cast a low amber glow from above, pooling warm light across the shoulders and the table surface while the corners of the frame fall away into deep brown. Style: lifestyle editorial. Mood: cozy, unhurried.',
            '[3:4] A medium shot of [trigger] walking through an autumn park, body turned three-quarters from the lens, head following the path ahead. Late-afternoon sun filters through thinning leaves overhead, dropping shifting patches of gold across the coat and the russet carpet of fallen leaves underfoot. Style: cinematic narrative. Mood: introspective, seasonal.',
            '[3:4] A medium shot of [trigger] in a denim jacket, leaning lightly against a brick wall in three-quarter pose, looking off down the street. The setting sun bounces off glass facades opposite, pushing a soft warm orange across the face and shoulder, while the wall behind already drifts toward the cool blue of early dusk. Style: street fashion editorial. Mood: confident, kinetic.',
            "[3:4] A medium shot of [trigger] in strict profile, seated by a rain-streaked window, gaze fixed somewhere beyond the glass. Cool grey daylight pours through from behind, throwing the face into a soft silhouette, with only a faint warm bounce from the room's interior catching the near cheek and the line of the lashes. Style: cinematic atmospheric. Mood: pensive, melancholic.",
            '[16:9] A wide full-body shot of [trigger] standing centered in an open snowy field, facing the camera directly, hands tucked into coat pockets. The low winter sun behind the lens grazes the snow with long pale-gold light, throwing a long cool-blue shadow across the white ground; distant pine-covered mountains hold the horizon in cold haze. Style: minimalist landscape portrait. Mood: still, expansive.',
            '[16:9] A wide full-body shot of [trigger] on a city bridge in three-quarter pose, leaning a forearm on the railing, looking out across the water. The sky has settled into deep cobalt blue while warm sodium streetlights and lit office windows speckle the scene, balancing cool overhead twilight with scattered golden practical accents along the figure. Style: cinematic urban. Mood: reflective, alive.',
            '[3:4] A wide full-body shot of [trigger] facing the camera in front of a richly tagged graffiti wall, arms relaxed at the sides. Diffuse midday light, bounced and softened by overcast cover, flattens the scene with even illumination, letting the saturated colors of the wall and the crisp outline of the figure read without competing with hard shadow. Style: documentary streetwear. Mood: bold, frontal.'
        ]
    }
];
