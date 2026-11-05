// На Windows в release-сборке не показывать консоль
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::VecDeque,
    error::Error,
    f32::consts::PI,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use eframe::egui::{self, Color32, Key, Stroke, Vec2};
use rodio::{source::SeekError, Decoder, OutputStream, OutputStreamHandle, Sink, Source};

// ───────── Настройки визуализатора ─────────
const WAVE_WINDOW: usize = 512; // сколько последних сэмплов канала показываем
const RAYS_PER_CHANNEL: usize = 256; // лучей на каждую полуокружность
const BASE_RADIUS: f32 = 0.6; // средний радиус (доля от максимального)
const WAVE_AMP: f32 = 1.; // размах волны (доля от максимального), умножается на GAIN
const GAIN: f32 = 2.0; // усиление сигнала (если тихо — увеличьте)
const CENTER_HOLE: f32 = 0.; // с какого радиуса начинаются лучи
const BUF_MAX: usize = 4096; // размер буфера кадров
const SMOOTH_TIME: f32 = 0.1; // сглаживание в секундах: больше = плавнее и медленнее, 0 = выключить

const TARGET_FPS: f32 = 30.0; // частота перерисовки визуализатора

type Shared = Arc<Mutex<VecDeque<[f32; 2]>>>;

// ───────────────── Перехватчик сэмплов ─────────────────
// Оборачивает источник звука и параллельно складывает стереокадры в общий буфер.
struct Tap<S: Source<Item = f32>> {
    inner: S,
    channels: u16,
    idx: u16,
    frame: [f32; 2],
    local: Vec<[f32; 2]>,
    shared: Shared,
}

impl<S: Source<Item = f32>> Tap<S> {
    fn new(inner: S, shared: Shared) -> Self {
        let channels = inner.channels().max(1);
        Self {
            inner,
            channels,
            idx: 0,
            frame: [0.0; 2],
            local: Vec::with_capacity(256),
            shared,
        }
    }
}

impl<S: Source<Item = f32>> Iterator for Tap<S> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let s = self.inner.next()?;
        if self.idx < 2 {
            self.frame[self.idx as usize] = s;
        }
        self.idx += 1;
        if self.idx >= self.channels {
            // моно: дублируем в оба канала
            if self.channels == 1 {
                self.frame[1] = self.frame[0];
            }
            self.local.push(self.frame);
            self.idx = 0;
            if self.local.len() >= 128 {
                if let Ok(mut buf) = self.shared.lock() {
                    buf.extend(self.local.iter().copied());
                    let extra = buf.len().saturating_sub(BUF_MAX);
                    buf.drain(..extra);
                }
                self.local.clear();
            }
        }
        Some(s)
    }
}

impl<S: Source<Item = f32>> Source for Tap<S> {
    fn current_frame_len(&self) -> Option<usize> {
        self.inner.current_frame_len()
    }
    fn channels(&self) -> u16 {
        self.inner.channels()
    }
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.inner.try_seek(pos)
    }
}

// ───────────────── Приложение ─────────────────
enum Act {
    Open,
    PlayIdx(usize),
    Toggle,
    Stop,
    Prev,
    Next,
}

struct App {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    sink: Option<Sink>,
    playlist: Vec<PathBuf>,
    current: Option<usize>,
    volume: f32,
    paused: bool,
    duration: Option<Duration>,
    samples: Shared,
    smooth: [Vec<f32>; 2], // сглаженные значения волны для каждого луча
    level: f32,            // сглаженная громкость для свечения
    accum: f32,            // накопленное время с последнего обновления анимации
    show_ui: bool,
    fullscreen: bool,
    error: Option<String>,
    seek_pos: Option<f32>,
}

impl App {
    fn new() -> Self {
        let (stream, handle) =
            OutputStream::try_default().expect("Не найдено аудиоустройство вывода");

        let mut app = Self {
            _stream: stream,
            handle,
            sink: None,
            playlist: Vec::new(),
            current: None,
            volume: 0.8,
            paused: false,
            duration: None,
            samples: Arc::new(Mutex::new(VecDeque::new())),
            smooth: [vec![0.0; RAYS_PER_CHANNEL], vec![0.0; RAYS_PER_CHANNEL]],
            level: 0.0,
            accum: 0.0,
            show_ui: true,
            fullscreen: false,
            error: None,
            seek_pos: None,
        };

        let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
        if !args.is_empty() {
            app.add_files(args);
        }
        app
    }

    fn add_files(&mut self, files: Vec<PathBuf>) {
        let was_empty = self.playlist.is_empty();
        let start = self.playlist.len();
        self.playlist.extend(files);
        if (was_empty || self.sink.is_none()) && start < self.playlist.len() {
            self.play_index(start);
        }
    }

    fn start(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        let file = BufReader::new(File::open(path)?);
        let decoder = Decoder::new(file)?;
        self.duration = decoder.total_duration();
        let source = Tap::new(decoder.convert_samples::<f32>(), self.samples.clone());
        let sink = Sink::try_new(&self.handle)?;
        sink.set_volume(self.volume);
        sink.append(source);
        self.sink = Some(sink);
        Ok(())
    }

    fn play_index(&mut self, idx: usize) {
        if let Some(s) = self.sink.take() {
            s.stop();
        }
        if let Ok(mut b) = self.samples.lock() {
            b.clear();
        }
        self.current = Some(idx);
        self.paused = false;
        self.seek_pos = None;
        let path = self.playlist[idx].clone();
        match self.start(&path) {
            Ok(()) => self.error = None,
            Err(e) => {
                self.sink = None;
                self.error = Some(format!("{}: {e}", path.display()));
            }
        }
    }

    fn stop(&mut self) {
        if let Some(s) = self.sink.take() {
            s.stop();
        }
        self.paused = false;
        self.seek_pos = None;
    }

    fn toggle(&mut self) {
        if let Some(s) = &self.sink {
            if self.paused {
                s.play();
            } else {
                s.pause();
            }
            self.paused = !self.paused;
        } else if let Some(i) = self.current {
            self.play_index(i);
        } else if !self.playlist.is_empty() {
            self.play_index(0);
        }
    }

    fn next(&mut self) {
        if let Some(i) = self.current {
            if i + 1 < self.playlist.len() {
                self.play_index(i + 1);
            } else {
                self.stop();
            }
        }
    }

    fn prev(&mut self) {
        if let Some(i) = self.current {
            self.play_index(i.saturating_sub(1));
        }
    }

    fn position(&self) -> f32 {
        self.sink
            .as_ref()
            .map(|s| s.get_pos().as_secs_f32())
            .unwrap_or(0.0)
    }

    fn apply(&mut self, act: Act) {
        match act {
            Act::Open => {
                if let Some(files) = rfd::FileDialog::new()
                    .add_filter("Аудио", &["mp3", "flac", "wav", "ogg", "m4a", "aac"])
                    .pick_files()
                {
                    self.add_files(files);
                }
            }
            Act::PlayIdx(i) => self.play_index(i),
            Act::Toggle => self.toggle(),
            Act::Stop => self.stop(),
            Act::Prev => self.prev(),
            Act::Next => self.next(),
        }
    }

    // Последние WAVE_WINDOW стереокадров (нули, если не играет)
    fn snapshot(&self) -> Vec<[f32; 2]> {
        let mut out = vec![[0.0f32; 2]; WAVE_WINDOW];
        if self.sink.is_some() && !self.paused {
            if let Ok(buf) = self.samples.lock() {
                let n = buf.len();
                if n >= WAVE_WINDOW {
                    for (o, f) in out.iter_mut().zip(buf.iter().skip(n - WAVE_WINDOW)) {
                        *o = *f;
                    }
                }
            }
        }
        out
    }

    // ───────── Визуализатор формы волны ─────────
    // Левый канал — верхняя полуокружность, правый — нижняя.
    // Каждый луч — один отсчёт формы волны: длина = BASE + сигнал * размах.
    fn draw_visual(&mut self, ui: &mut egui::Ui) {
        // 1. Обновление анимации — строго с частотой TARGET_FPS
        let step = 1.0 / TARGET_FPS;
        self.accum += ui.input(|i| i.stable_dt).min(0.1);
        if self.accum >= step {
            self.accum -= step;
            if self.accum > step {
                self.accum = 0.0; // не копим «долг» после лагов
            }

            let alpha = if SMOOTH_TIME > 0.0 {
                1.0 - (-step / SMOOTH_TIME).exp()
            } else {
                1.0
            };
            let wave = self.snapshot();

            let rms = (wave.iter().map(|f| f[0] * f[0] + f[1] * f[1]).sum::<f32>()
                / (2.0 * wave.len() as f32))
                .sqrt();
            let target_level = (rms * GAIN * 2.0).clamp(0.0, 1.0);
            self.level += (target_level - self.level) * alpha;

            for ch in 0..2 {
                for i in 0..RAYS_PER_CHANNEL {
                    let t = i as f32 / (RAYS_PER_CHANNEL - 1) as f32;
                    let pos = t * (WAVE_WINDOW - 1) as f32;
                    let i0 = pos.floor() as usize;
                    let i1 = (i0 + 1).min(WAVE_WINDOW - 1);
                    let f = pos - i0 as f32;
                    let target = wave[i0][ch] * (1.0 - f) + wave[i1][ch] * f;
                    let sm = &mut self.smooth[ch][i];
                    *sm += (target - *sm) * alpha;
                }
            }
        }

        // 2. Отрисовка — каждый кадр, из текущего состояния
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, Color32::BLACK);

        let c = rect.center();
        let max_r = rect.width().min(rect.height()) * 0.5 * 0.95;
        let hole = max_r * CENTER_HOLE;
        let level = self.level;

        // Свечение в центре

        for ch in 0..2 {
            for i in 0..RAYS_PER_CHANNEL {
                let t = i as f32 / (RAYS_PER_CHANNEL - 1) as f32;
                let s = self.smooth[ch][i];

                // верх (канал 0): от левой точки через верх к правой
                // низ (канал 1): от левой точки через низ к правой
                let a = if ch == 0 { PI + t * PI } else { PI - t * PI };
                let dir = Vec2::new(a.cos(), a.sin());

                let r = (BASE_RADIUS + s * GAIN * WAVE_AMP * 0.5).clamp(0.05, 1.0) * max_r;
                let k = r / max_r;
                let color = Color32::from_rgb(255, 250, (100.0 + 140.0 * k).min(255.0) as u8);
                p.line_segment([c + dir * hole, c + dir * r], Stroke::new(1.0, color));
            }
        }
    }
}

fn fmt_time(secs: f32) -> String {
    let s = secs as u32;
    format!("{:02}:{:02}", s / 60, s % 60)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut act: Option<Act> = None;

        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.add_files(dropped);
        }

        if ctx.input(|i| i.key_pressed(Key::Space)) {
            act = Some(Act::Toggle);
        }
        if ctx.input(|i| i.key_pressed(Key::F)) {
            self.show_ui = !self.show_ui;
        }
        if ctx.input(|i| i.key_pressed(Key::F11)) {
            self.fullscreen = !self.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
        }

        if !self.paused {
            if let Some(s) = &self.sink {
                if s.empty() {
                    act = Some(Act::Next);
                }
            }
        }

        if self.show_ui {
            egui::SidePanel::left("playlist")
                .default_width(240.0)
                .show(ctx, |ui| {
                    ui.heading("Плейлист");
                    if ui.button("Открыть файлы…").clicked() {
                        act = Some(Act::Open);
                    }
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (i, path) in self.playlist.iter().enumerate() {
                            let name = path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                            let selected = self.current == Some(i);
                            if ui.selectable_label(selected, name).clicked() {
                                act = Some(Act::PlayIdx(i));
                            }
                        }
                    });
                });

            egui::TopBottomPanel::bottom("controls").show(ctx, |ui| {
                ui.add_space(4.0);
                if let Some(e) = &self.error {
                    ui.colored_label(Color32::LIGHT_RED, e);
                } else if let Some(i) = self.current {
                    let name = self.playlist[i]
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    ui.label(name);
                } else {
                    ui.label("Перетащите файлы в окно или нажмите «Открыть файлы…»");
                }

                let dur = self.duration.map(|d| d.as_secs_f32()).unwrap_or(0.0);
                let pos = self.position();
                ui.horizontal(|ui| {
                    ui.label(fmt_time(self.seek_pos.unwrap_or(pos)));
                    if dur > 0.0 {
                        let mut v = self.seek_pos.unwrap_or(pos).min(dur);
                        ui.spacing_mut().slider_width = (ui.available_width() - 70.0).max(100.0);
                        let r = ui.add(
                            egui::Slider::new(&mut v, 0.0..=dur)
                                .show_value(false)
                                .trailing_fill(true),
                        );
                        if r.changed() {
                            self.seek_pos = Some(v);
                        }
                        if r.drag_stopped() || (r.changed() && !r.dragged()) {
                            if let Some(s) = &self.sink {
                                if let Err(e) = s.try_seek(Duration::from_secs_f32(v)) {
                                    self.error = Some(format!("Перемотка не удалась: {e:?}"));
                                }
                            }
                            self.seek_pos = None;
                        }
                        ui.label(fmt_time(dur));
                    } else {
                        ui.label("--:--");
                    }
                });

                ui.horizontal(|ui| {
                    if ui.button("Пред.").clicked() {
                        act = Some(Act::Prev);
                    }
                    let label = if self.sink.is_some() && !self.paused {
                        "Пауза"
                    } else {
                        "Играть"
                    };
                    if ui.button(label).clicked() {
                        act = Some(Act::Toggle);
                    }
                    if ui.button("Стоп").clicked() {
                        act = Some(Act::Stop);
                    }
                    if ui.button("След.").clicked() {
                        act = Some(Act::Next);
                    }
                    ui.separator();
                    ui.label("Громкость");
                    if ui
                        .add(egui::Slider::new(&mut self.volume, 0.0..=1.0).show_value(false))
                        .changed()
                    {
                        if let Some(s) = &self.sink {
                            s.set_volume(self.volume);
                        }
                    }
                    ui.separator();
                    ui.weak("F — только визуализатор, F11 — полный экран");
                });
                ui.add_space(4.0);
            });
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::BLACK))
            .show(ctx, |ui| {
                self.draw_visual(ui);
            });

        if let Some(a) = act {
            self.apply(a);
        }

        ctx.request_repaint(); // окно перерисовывается в ритме монитора
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 700.0])
            .with_title("Rust Player")
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "Rust Player",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
