use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

const PI: f32 = 3.1415926535;

#[derive(Debug, Copy, Clone)]
struct LightSource {
    position: Vector3<f32>,
    intensity: Vector3<f32>,
}

#[derive(Debug, Copy, Clone)]
struct AmbientLight {
    intensity: Vector3<f32>,
}

#[derive(Copy, Clone)]
enum Projection {
    Orthogonal,
    Perspective,
}

#[derive(Debug, Copy, Clone)]
struct Ray {
    direction: Vector3<f32>,
    origin: Vector3<f32>,
}

#[derive(Debug, Copy, Clone)]
struct Coordinate<T> {
    u: Vector3<T>,
    v: Vector3<T>,
    w: Vector3<T>,
}

#[derive(Debug, Copy, Clone)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug, Copy, Clone)]
struct Material {
    diffuse: Color,
    albedo: Color,
    specular: Color,
    mirror: Vector3<f32>,
    specular_p: f32,
}

#[derive(Clone)]
struct Canvas {
    width: usize,
    height: usize,
    data: Vec<Color>,
}

#[derive(Copy, Clone)]
struct Camera {
    focal: f32,
    position: Vector3<f32>,
    projection: Projection,
}

#[derive(Copy, Clone, Debug)]
struct Vector2<T> {
    x: T,
    y: T,
}

#[derive(Debug, Clone, Copy)]
struct Vector3<T> {
    x: T,
    y: T,
    z: T,
}

#[derive(Debug, Clone, Copy)]
struct Vector4<T> {
    w: T,
    x: T,
    y: T,
    z: T,
}

#[derive(Debug, Clone, Copy)]
struct Quaternion<T> {
    w: T,
    v: Vector3<T>,
}

impl Vector3<f32> {
    fn norm(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize(self) -> Vector3<f32> {
        self / self.norm()
    }
}

impl Color {
    fn intensity(self, int: Vector3<f32>) -> Color {
        Color {
            r: (int.x * self.r as f32) as u8,
            g: (int.y * self.g as f32) as u8,
            b: (int.z * self.b as f32) as u8,
        }
    }

    fn gamma_correction(&self, gamma: f32) -> Color {
        let exp = 1. / gamma;
        Color {
            r: (self.r as f32).powf(exp).round() as u8,
            g: (self.g as f32).powf(exp).round() as u8,
            b: (self.b as f32).powf(exp).round() as u8,
        }
    }

    fn as_buf(&self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }
}

#[derive(Debug, Copy, Clone)]
struct Sphere {
    center: Vector3<f32>,
    radius: f32,
    material: Material,
}

#[derive(Debug, Copy, Clone)]
struct Plane {
    normal: Vector3<f32>,
    position: Vector3<f32>,
    material: Material,
}

impl<T> Neg for Vector3<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vector3::<T> {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T> Add for Vector3<T>
where
    T: Add<Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::<T> {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Add for Color {
    type Output = Self;
    fn add(self, rhs: Color) -> Self::Output {
        Color {
            r: std::cmp::min(255, self.r as u16 + rhs.r as u16) as u8,
            g: std::cmp::min(255, self.g as u16 + rhs.g as u16) as u8,
            b: std::cmp::min(255, self.b as u16 + rhs.b as u16) as u8,
        }
    }
}

impl<T> Sub for Vector3<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::<T> {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Mul for Vector3<T>
where
    T: Mul<Output = T> + Add<Output = T>,
{
    type Output = T;
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl<T> Mul for Quaternion<T>
where
    T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy,
{
    type Output = Quaternion<T>;
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        let (w1, x1, y1, z1) = (self.w, self.v.x, self.v.y, self.v.z);
        let (w2, x2, y2, z2) = (rhs.w, rhs.v.x, rhs.v.y, rhs.v.z);

        Quaternion::<T> {
            w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            v: Vector3::<T> {
                x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            },
        }
    }
}

impl Quaternion<f32> {
    fn from_vector3f(vector3: Vector3<f32>) -> Quaternion<f32> {
        Quaternion::<f32> { w: 0., v: vector3 }
    }

    fn from_euler(axis: Vector3<f32>, degree: f32) -> Quaternion<f32> {
        let radian: f32 = PI * degree / 180. / 2.;
        let (sin, cos) = radian.sin_cos();
        Quaternion::<f32> {
            w: cos,
            v: sin * axis,
        }
    }

    fn conjugate(self) -> Quaternion<f32> {
        Quaternion::<f32> {
            w: self.w,
            v: -self.v,
        }
    }
}

impl<T: Mul<f32, Output = T>> Mul<Vector3<T>> for f32 {
    type Output = Vector3<T>;
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::<T> {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
        }
    }
}

impl Div<f32> for Vector3<f32> {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Vector3::<f32> {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Vector3<f32> {
    fn rotate(&self, quaternion: Quaternion<f32>) -> Self {
        let rotated_quaternion =
            quaternion * Quaternion::<f32>::from_vector3f(*self) * quaternion.conjugate();
        rotated_quaternion.v
    }

    fn rotate_world(&self, center: Vector3<f32>, quaternion: Quaternion<f32>) -> Self {
        (*self - center).rotate(quaternion) + center
    }
}

trait Hitable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<f32>;
    fn normal(&self, p: Vector3<f32>) -> Vector3<f32>;
}

trait Lightable {
    fn get_material(&self) -> Material;
}

impl Lightable for Sphere {
    fn get_material(&self) -> Material {
        self.material
    }
}

impl Lightable for Plane {
    fn get_material(&self) -> Material {
        self.material
    }
}

impl Hitable for Sphere {
    fn normal(&self, p: Vector3<f32>) -> Vector3<f32> {
        (p - self.center) / self.radius
    }

    fn hit(&self, ray: &Ray, t_min: f32, _t_max: f32) -> Option<f32> {
        let a = ray.direction * ray.direction;
        let b = 2. * ray.direction * (ray.origin - self.center);
        let c = (ray.origin - self.center) * (ray.origin - self.center) - self.radius * self.radius;

        let delta = b * b - 4. * a * c;
        if delta < 0. {
            return None;
        } else {
            let delta = delta.sqrt();
            let x1 = (-b + delta) / (2. * a);
            let x2 = (-b - delta) / (2. * a);

            if x1 < t_min && x2 < t_min {
                None
            } else {
                Some(if x1 < t_min && x2 >= t_min {
                    x2
                } else if x2 < t_min && x1 >= t_min {
                    x1
                } else {
                    x1.min(x2)
                })
            }
        }
    }
}

impl Hitable for Plane {
    fn normal(&self, _p: Vector3<f32>) -> Vector3<f32> {
        self.normal
    }

    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<f32> {
        let t = self.normal * (self.position - ray.origin) / (self.normal * ray.direction);
        if t_min < t && t < t_max {
            Some(t)
        } else {
            None
        }
    }
}

trait Renderable: Hitable + Lightable + std::fmt::Debug {}

impl Renderable for Plane {}
impl Renderable for Sphere {}

struct Scene<'a> {
    objects: &'a Vec<Box<dyn Renderable>>,
    ambient: AmbientLight,
    lights: &'a Vec<LightSource>,
}

fn ray(
    pixel: Vector2<usize>,
    camera: Camera,
    coordinate: Coordinate<f32>,
    canvas: &Canvas,
    plane: Vector2<f32>,
) -> Ray {
    let d: f32 = camera.focal;
    let (i, j) = (pixel.x as f32, pixel.y as f32);
    let (width, height) = (canvas.width as f32, canvas.height as f32);

    let l: f32 = -plane.x / 2.;
    let r: f32 = -l;
    let b: f32 = -plane.y / 2.;
    let t: f32 = -b;
    let u: f32 = l + (r - l) * (i - 0.5) / width;
    let v: f32 = b + (t - b) * (j - 0.5) / height;

    let (direction, origin) = match camera.projection {
        Projection::Perspective => (
            -d * coordinate.w + u * coordinate.u + v * coordinate.v,
            camera.position,
        ),
        Projection::Orthogonal => (
            -coordinate.w,
            camera.position + u * coordinate.u + v * coordinate.v,
        ),
    };

    Ray { direction, origin }
}

const MAX_RECURSION_DEPTH: usize = 10;

fn ray_color(r: Ray, scene: &Scene, t_min: f32, t_max: f32, depth: usize) -> Color {
    let objects = scene.objects;
    let ambient = scene.ambient;
    let lights = scene.lights;
    let (min_t, hit_obj) = hit_scene(r, objects, t_min, t_max);

    if let Some(obj) = hit_obj {
        let p = r.origin + min_t.unwrap() * r.direction;
        let v = (-p).normalize();
        let n = obj.normal(p).normalize();
        let mat = obj.get_material();
        let ambient_color = mat.albedo.intensity(ambient.intensity);
        let mut color = ambient_color;

        for light in lights {
            let l = (light.position - p).normalize();
            if let (None, None) = hit_scene(
                Ray {
                    origin: p,
                    direction: l,
                },
                objects,
                1e-2,
                std::f32::INFINITY,
            ) {
                let h = (v + l).normalize();
                let diffuse_intensity = (l * n).max(0.) * light.intensity;
                let diffuse_color = mat.diffuse.intensity(diffuse_intensity);
                let specular_intensity = (h * n).max(0.).powf(mat.specular_p) * light.intensity;
                let specular_color = mat.specular.intensity(specular_intensity);
                color = color + diffuse_color + specular_color;

                if depth < MAX_RECURSION_DEPTH {
                    let mirror_intensity = mat.mirror;
                    const eps: f32 = 1e-5;
                    if mirror_intensity.x < eps
                        && mirror_intensity.y < eps
                        && mirror_intensity.z < eps
                    {
                        continue;
                    }

                    let d = r.direction.normalize();

                    let rr = Ray {
                        origin: p,
                        direction: d - 2. * (d * n) * n,
                    };

                    color = color
                        + ray_color(rr, scene, 1e-2, std::f32::INFINITY, depth + 1)
                            .intensity(mirror_intensity);
                }
            }
        }
        color
    } else {
        // No hit, background color
        Color { r: 0, g: 0, b: 0 }
    }
}

fn hit_scene(
    r: Ray,
    objects: &Vec<Box<dyn Renderable>>,
    t_min: f32,
    t_max: f32,
) -> (Option<f32>, Option<&Box<dyn Renderable>>) {
    let mut min_t: Option<f32> = None;
    let mut hit_obj: Option<&Box<dyn Renderable>> = None;

    for object in objects {
        if let Some(t) = object.hit(&r, t_min, t_max) {
            let update: bool;
            if let Some(old_t) = min_t {
                update = old_t > t;
            } else {
                update = true;
            }

            if update {
                min_t = Some(t);
                hit_obj = Some(object);
            }
        }
    }

    (min_t, hit_obj)
}

fn render(
    camera: Camera,
    coordinate: Coordinate<f32>,
    canvas: &mut Canvas,
    plane: Vector2<f32>,
    scene: &Scene,
) {
    let mut stderr = std::io::stderr();
    for y in 0..canvas.height {
        write!(
            &mut stderr,
            "\rRendering Line {}/{}\r",
            y + 1,
            canvas.height,
        )
        .expect("Could not write to stderr");
        for x in 0..canvas.width {
            let index = y * canvas.height + x;
            let pixel = Vector2 {
                x,
                y: canvas.height - y,
            };
            let r = ray(pixel, camera, coordinate, canvas, plane);
            canvas.data[index] = ray_color(r, scene, 1., std::f32::INFINITY, 0);
        }
    }
}

fn main() -> std::io::Result<()> {
    let plane = Vector2::<f32> { x: 0.1, y: 0.1 };

    let green = Color {
        r: 0,
        g: 200,
        b: 20,
    };

    let white = Color {
        r: 255,
        g: 255,
        b: 255,
    };

    let red = Color {
        r: 200,
        g: 0,
        b: 20,
    };

    let blue = Color {
        r: 0,
        g: 85,
        b: 255,
    };

    let mirror = Vector3::<f32> {
        x: 0.2,
        y: 0.2,
        z: 0.2,
    };

    let green_material = Material {
        albedo: green,
        diffuse: green,
        specular: white,
        mirror: mirror,
        specular_p: 25.,
    };

    let red_material = Material {
        albedo: red,
        diffuse: red,
        specular: white,
        mirror: mirror,
        specular_p: 1.,
    };

    let blue_material = Material {
        albedo: blue,
        diffuse: blue,
        specular: white,
        mirror: mirror,
        specular_p: 150.,
    };

    let white_material = Material {
        albedo: white,
        diffuse: white,
        specular: white,
        mirror: mirror,
        specular_p: 100.,
    };

    let ambient = AmbientLight {
        intensity: Vector3::<f32> {
            x: 0.2,
            y: 0.2,
            z: 0.2,
        },
    };

    let light_1 = LightSource {
        position: Vector3::<f32> {
            x: -20.,
            y: 27.,
            z: -30.,
        },
        intensity: Vector3::<f32> {
            x: 0.4,
            y: 0.4,
            z: 0.4,
        },
    };

    let light_2 = LightSource {
        position: Vector3::<f32> {
            x: -2.,
            y: 20.,
            z: 10.,
        },
        intensity: Vector3::<f32> {
            x: 0.3,
            y: 0.3,
            z: 0.3,
        },
    };

    let light_3 = LightSource {
        position: Vector3::<f32> {
            x: 15.,
            y: 25.,
            z: -60.,
        },
        intensity: Vector3::<f32> {
            x: 0.3,
            y: 0.3,
            z: 0.3,
        },
    };

    let scene = Scene {
        objects: &vec![
            Box::new(Sphere {
                center: Vector3::<f32> {
                    x: -0.17,
                    y: 0.06,
                    z: -10.,
                },
                radius: 0.16,
                material: blue_material,
            }),
            Box::new(Sphere {
                center: Vector3::<f32> {
                    x: 0.17,
                    y: 0.05,
                    z: -10.5,
                },
                radius: 0.15,
                material: green_material,
            }),
            Box::new(Plane {
                position: Vector3::<f32> {
                    x: 0.,
                    y: -0.10,
                    z: -10.,
                },
                normal: (Vector3::<f32> {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                })
                .rotate(Quaternion::from_euler(
                    Vector3::<f32> {
                        x: 1.,
                        y: 0.,
                        z: 0.,
                    },
                    0.,
                )),
                material: white_material,
            }),
        ],
        ambient,
        lights: &vec![light_1, light_2, light_3],
    };

    let args: Vec<String> = env::args().collect();

    let frames: usize = (&args[1]).parse().expect("invalid frames");
    let resolution: usize = (&args[2]).parse().expect("invalid resolution");
    let dir = &args[3];
    let width: usize = resolution;
    let height: usize = resolution;

    for frame in 0..frames {
        let degree: f32 = frame as f32;

        let u = Vector3::<f32> {
            x: 1.,
            y: 0.,
            z: 0.,
        };

        let v = Vector3::<f32> {
            x: 0.,
            y: 1.,
            z: 0.,
        };

        let w = Vector3::<f32> {
            x: 0.,
            y: 0.,
            z: 1.,
        };

        let quaternion = Quaternion::<f32>::from_euler(v, degree);

        let u = u.rotate(quaternion);

        let v = v.rotate(quaternion);

        let w = w.rotate(quaternion);

        let quaternion2 = Quaternion::<f32>::from_euler(u, -5.);

        let u = u.rotate(quaternion2);

        let v = v.rotate(quaternion2);

        let w = w.rotate(quaternion2);

        let coordinate = Coordinate::<f32> { u, v, w };

        let center = Vector3::<f32> {
            x: 0.,
            y: 0.,
            z: -10.25,
        };
        let camera = Camera {
            focal: 1.,
            position: (Vector3::<f32> {
                x: 0.,
                y: 0.,
                z: 0.,
            })
            .rotate_world(center, quaternion)
            .rotate_world(center, quaternion2),
            projection: Projection::Perspective,
        };

        let canvas = &mut Canvas {
            width,
            height,
            data: vec![Color { r: 0, g: 0, b: 0 }; width * height],
        };

        render(camera, coordinate, canvas, plane, &scene);
        let path = format!("{}/{:03}.ppm", dir, frame);
        println!("Writing to {}...", path);
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "P6")?;
        writeln!(writer, "{} {}", canvas.width, canvas.height)?;
        writeln!(writer, "255")?;
        for rgb in &canvas.data {
            let pixel = rgb;
            writer.write(&pixel.as_buf())?;
        }
        writer.flush()?;
        println!("Finished.");
    }
    Ok(())
}
