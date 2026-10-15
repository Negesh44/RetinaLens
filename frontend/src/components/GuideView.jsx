import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Camera,
  CheckCircle2,
  FileText,
  ScanEye,
  Stethoscope,
  Upload,
} from 'lucide-react';

const STEPS = [
  {
    icon: Camera,
    title: 'Get a photo of the retina',
    lead: 'A retinal (fundus) photograph is taken with a special eye camera. A phone photo of the eye will not work: it shows the front of the eye, not the retina.',
    groups: [
      {
        heading: 'Where to go',
        items: [
          'An optometrist or optical store with a retinal camera',
          'An ophthalmologist (eye doctor) or eye hospital',
          'A diabetic eye screening clinic or a hospital eye department',
        ],
      },
      {
        heading: 'What happens',
        items: [
          'You sit at the camera with your chin on a rest and look at a small light.',
          'A bright flash takes the photo. It is painless and takes a few minutes per eye.',
          'Sometimes eye drops widen the pupil first. Vision is blurry and sensitive to light for 4–6 hours, so bring sunglasses and do not drive until it clears.',
        ],
      },
      {
        heading: 'What to ask for',
        items: [
          'A digital copy of the photo (JPEG or PNG), one image per eye.',
          'Sharp, well-lit images showing the optic disc and the macula.',
          'The original file, not a screenshot or a photo of a printed picture.',
        ],
      },
    ],
  },
  {
    icon: Upload,
    title: 'Upload it to RetinaLens',
    lead: 'Open the Screening tab and add the photo.',
    groups: [
      {
        heading: 'Steps',
        items: [
          'Optional: enter a patient ID and age.',
          'Choose which eye the photo shows: right (OD) or left (OS).',
          'Drag the photo into the upload box, or click it to choose the file.',
          'Press Begin AI Screening. The analysis takes a few seconds.',
        ],
      },
      {
        heading: 'If the photo is rejected',
        items: [
          'The quality check rejects photos that are blurry, too dark or badly framed, because they cannot be read reliably.',
          'Ask the clinic to retake the photo, then upload the new one.',
        ],
      },
    ],
  },
  {
    icon: ScanEye,
    title: 'Read the result',
    lead: 'The report opens automatically after the analysis.',
    groups: [
      {
        heading: 'What you will see',
        items: [
          'Primary diagnosis: the condition the AI found most likely, or "No Abnormality Detected".',
          'Urgency: how soon to see an eye doctor (see the table below).',
          'AI confidence, plus how reliable the AI was for that condition on test images.',
          'Detected conditions: only the conditions that were found are listed.',
          'Click to Inspect: click the photo to see which areas the AI relied on.',
        ],
      },
    ],
  },
  {
    icon: FileText,
    title: 'Use the report',
    lead: 'The AI result is a screening aid, not a diagnosis. Always confirm it with an eye doctor.',
    groups: [
      {
        heading: 'What to do',
        items: [
          'Press Download PDF Report and save it.',
          'Book an appointment with an ophthalmologist, as soon as the urgency label says.',
          'Bring the PDF report and the original retina photos to the appointment.',
          'The doctor will examine the eye and may order tests such as OCT, eye pressure or visual fields.',
          'The report contains patient details: share it only with the patient and their doctors.',
        ],
      },
    ],
  },
];

const GRID_COLS = { 1: '', 2: 'md:grid-cols-2', 3: 'md:grid-cols-3' };

const URGENCY = [
  { label: 'Emergency · Same day', action: 'Go to an eye hospital or emergency department today.', className: 'bg-red-500/15 text-red-300 ring-red-500/40' },
  { label: 'Urgent · 24–48 hours', action: 'See an ophthalmologist within 1–2 days.', className: 'bg-rose-500/15 text-rose-300 ring-rose-500/40' },
  { label: 'Prompt · 2–4 weeks', action: 'Book an ophthalmologist appointment within the next few weeks.', className: 'bg-amber-500/15 text-amber-300 ring-amber-500/40' },
  { label: 'Routine follow-up', action: 'No signs found. Keep regular eye check-ups, at least yearly with diabetes.', className: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/40' },
  { label: 'Abnormality, type not identified', action: 'Something looks abnormal. Get a full eye examination soon.', className: 'bg-amber-500/15 text-amber-300 ring-amber-500/40' },
  { label: 'Retake photo', action: 'The photo could not be read. Have it retaken and upload again.', className: 'bg-slate-700/40 text-slate-300 ring-slate-600/50' },
];

const SPECIALISTS = [
  ['Diabetic retinopathy', 'Ophthalmologist (retina specialist), plus your diabetes doctor to control sugar and blood pressure'],
  ['Glaucoma', 'Ophthalmologist (glaucoma clinic): eye pressure, optic nerve scan and visual field tests'],
  ['Macular degeneration (AMD)', 'Ophthalmologist (retina specialist)'],
  ['Retinal vein occlusion', 'Ophthalmologist (retina specialist), plus your physician to check blood pressure, sugar and cholesterol'],
  ['Hypertensive retinopathy', 'Ophthalmologist, plus your physician to control blood pressure'],
  ['Optic disc edema', 'Ophthalmologist urgently: a swollen optic disc may need brain or nerve tests'],
  ['CSC, macular hole / ERM, chorioretinitis', 'Ophthalmologist (retina specialist)'],
  ['Pathologic myopia', 'Ophthalmologist, to check for complications behind the high short-sightedness'],
];

export default function GuideView({ onStart }) {
  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <header className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl sm:p-8">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-sky-400">How it works</p>
        <h1 className="mt-1 font-heading text-2xl font-bold text-white sm:text-3xl">From retina photo to eye doctor, in 4 steps</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          RetinaLens checks a retina photo for 10 common eye conditions and tells you how soon to see an eye doctor.
          It does not replace an eye examination.
        </p>
        <ol className="mt-5 flex flex-wrap gap-2">
          {STEPS.map((s, i) => (
            <li key={s.title}>
              <a
                href={`#step-${i + 1}`}
                className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-500"
              >
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-sky-500/20 font-mono text-[11px] font-bold text-sky-300">{i + 1}</span>
                {s.title}
              </a>
            </li>
          ))}
        </ol>
      </header>

      <section className="flex items-start gap-3 rounded-2xl border border-red-500/40 bg-red-500/10 p-5">
        <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-400" />
        <div className="text-sm text-slate-200">
          <p className="font-semibold text-red-300">Go to an eye emergency right away, whatever the AI says, if you have:</p>
          <p className="mt-1 leading-relaxed text-slate-300">
            sudden loss or blurring of vision, a shadow or curtain over part of your sight, a sudden shower of new floaters or
            flashes of light, or severe eye pain or headache with vision changes.
          </p>
        </div>
      </section>

      <ol className="relative space-y-5">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <li key={step.title} id={`step-${i + 1}`} className="scroll-mt-24 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg sm:p-6">
              <div className="flex items-start gap-4">
                <div className="flex flex-shrink-0 flex-col items-center">
                  <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-sky-500/15 text-sky-400 ring-1 ring-inset ring-sky-500/30">
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className="mt-1.5 font-mono text-[11px] font-semibold text-slate-500">STEP {i + 1}</span>
                </div>
                <div className="min-w-0 flex-1">
                  <h2 className="font-heading text-lg font-bold text-white">{step.title}</h2>
                  <p className="mt-1 text-sm leading-relaxed text-slate-400">{step.lead}</p>
                  <div className={`mt-4 grid gap-4 ${GRID_COLS[step.groups.length] ?? 'md:grid-cols-3'}`}>
                    {step.groups.map((g) => (
                      <div key={g.heading} className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">{g.heading}</h3>
                        <ul className="mt-2 space-y-1.5">
                          {g.items.map((item) => (
                            <li key={item} className="flex gap-2 text-sm leading-relaxed text-slate-200">
                              <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-500" />
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ol>

      <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg sm:p-6">
        <h2 className="font-heading text-lg font-bold text-white">What the urgency label means</h2>
        <p className="mt-1 text-sm text-slate-400">Every report shows one of these labels next to the diagnosis.</p>
        <ul className="mt-4 divide-y divide-slate-800">
          {URGENCY.map((u) => (
            <li key={u.label} className="flex flex-col gap-1.5 py-3 sm:flex-row sm:items-center sm:gap-4">
              <span className={`inline-flex w-fit flex-shrink-0 rounded-md px-2.5 py-1 text-xs font-semibold ring-1 ring-inset sm:w-56 ${u.className}`}>{u.label}</span>
              <span className="text-sm text-slate-200">{u.action}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg sm:p-6">
        <div className="flex items-center gap-2">
          <Stethoscope className="h-5 w-5 text-sky-400" />
          <h2 className="font-heading text-lg font-bold text-white">Who to see for each condition</h2>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="border-b border-slate-800 text-xs uppercase tracking-wider text-slate-400">
              <tr>
                <th className="py-2 pr-4 font-semibold">Condition found</th>
                <th className="py-2 font-semibold">Who to see</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {SPECIALISTS.map(([condition, who]) => (
                <tr key={condition}>
                  <td className="py-2.5 pr-4 font-medium text-white">{condition}</td>
                  <td className="py-2.5 text-slate-300">{who}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500">
          RetinaLens does not check for retinal detachment, eye tumours, diabetic macular edema or retinitis pigmentosa.
          A normal result does not rule out eye disease, so keep regular eye check-ups.
        </p>
      </section>

      <div className="flex justify-center pb-4">
        <button
          type="button"
          onClick={onStart}
          className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-6 py-3 text-sm font-semibold text-white shadow-lg transition-colors hover:bg-sky-500"
        >
          Start a screening <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
