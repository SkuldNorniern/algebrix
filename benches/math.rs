//! Benchmarks for the hot paths, with glam as the reference implementation.
//!
//! Run with `cargo bench`. Each group benches the same operation in
//! algebrix and glam so the report shows them side by side.
//! Compare feature sets with e.g.
//! `cargo bench --no-default-features --features std,scalar-math`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use algebrix::{Mat4, Quat, Vec3, Vec4};

fn vec3_ops(c: &mut Criterion) {
    let a = Vec3::new(1.2, -3.4, 5.6);
    let b = Vec3::new(-7.8, 9.0, 1.2);
    let ga = glam::Vec3::new(1.2, -3.4, 5.6);
    let gb = glam::Vec3::new(-7.8, 9.0, 1.2);

    let mut g = c.benchmark_group("vec3_dot");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a).dot(black_box(b)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga).dot(black_box(gb)))
    });
    g.finish();

    let mut g = c.benchmark_group("vec3_cross");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a).cross(black_box(b)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga).cross(black_box(gb)))
    });
    g.finish();

    let mut g = c.benchmark_group("vec3_length");
    g.bench_function("algebrix", |bench| bench.iter(|| black_box(a).length()));
    g.bench_function("glam", |bench| bench.iter(|| black_box(ga).length()));
    g.finish();

    let mut g = c.benchmark_group("vec3_normalize");
    g.bench_function("algebrix", |bench| bench.iter(|| black_box(a).normalize()));
    g.bench_function("glam", |bench| bench.iter(|| black_box(ga).normalize()));
    g.finish();

    let mut g = c.benchmark_group("vec3_lerp");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a).lerp(black_box(b), black_box(0.25)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga).lerp(black_box(gb), black_box(0.25)))
    });
    g.finish();
}

fn vec4_ops(c: &mut Criterion) {
    let a = Vec4::new(1.2, -3.4, 5.6, -7.8);
    let b = Vec4::new(9.0, 1.2, -3.4, 5.6);
    let ga = glam::Vec4::new(1.2, -3.4, 5.6, -7.8);
    let gb = glam::Vec4::new(9.0, 1.2, -3.4, 5.6);

    let mut g = c.benchmark_group("vec4_dot");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a).dot(black_box(b)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga).dot(black_box(gb)))
    });
    g.finish();

    let mut g = c.benchmark_group("vec4_add");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a) + black_box(b))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga) + black_box(gb))
    });
    g.finish();

    let mut g = c.benchmark_group("vec4_mul");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a) * black_box(b))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga) * black_box(gb))
    });
    g.finish();

    let mut g = c.benchmark_group("vec4_normalize");
    g.bench_function("algebrix", |bench| bench.iter(|| black_box(a).normalize()));
    g.bench_function("glam", |bench| bench.iter(|| black_box(ga).normalize()));
    g.finish();
}

fn mat4_ops(c: &mut Criterion) {
    let rot = Mat4::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7);
    let srt = Mat4::from_scale_rotation_translation(
        Vec3::new(1.5, 2.0, 0.5),
        Quat::from_axis_angle(Vec3::Y, 0.3),
        Vec3::new(4.0, 5.0, 6.0),
    );
    let v4 = Vec4::new(1.0, 2.0, 3.0, 1.0);
    let v3 = Vec3::new(1.0, 2.0, 3.0);

    let grot = glam::Mat4::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    let gsrt = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(1.5, 2.0, 0.5),
        glam::Quat::from_axis_angle(glam::Vec3::Y, 0.3),
        glam::Vec3::new(4.0, 5.0, 6.0),
    );
    let gv4 = glam::Vec4::new(1.0, 2.0, 3.0, 1.0);
    let gv3 = glam::Vec3::new(1.0, 2.0, 3.0);

    let mut g = c.benchmark_group("mat4_mul_mat4");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(rot) * black_box(srt))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(grot) * black_box(gsrt))
    });
    g.finish();

    let mut g = c.benchmark_group("mat4_mul_affine");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(rot).mul_affine(black_box(srt)))
    });
    g.finish();

    let mut g = c.benchmark_group("mat4_mul_vec4");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(rot).mul_vec4(black_box(v4)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(grot).mul_vec4(black_box(gv4)))
    });
    g.finish();

    let mut g = c.benchmark_group("mat4_transform_point3");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(srt).transform_point3(black_box(v3)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(gsrt).transform_point3(black_box(gv3)))
    });
    g.finish();

    let mut g = c.benchmark_group("mat4_transpose");
    g.bench_function("algebrix", |bench| bench.iter(|| black_box(srt).transpose()));
    g.bench_function("glam", |bench| bench.iter(|| black_box(gsrt).transpose()));
    g.finish();

    let mut g = c.benchmark_group("mat4_inverse");
    g.bench_function("algebrix", |bench| bench.iter(|| black_box(srt).inverse()));
    g.bench_function("glam", |bench| bench.iter(|| black_box(gsrt).inverse()));
    g.finish();

    let mut g = c.benchmark_group("mat4_determinant");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(srt).determinant())
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(gsrt).determinant())
    });
    g.finish();
}

fn quat_ops(c: &mut Criterion) {
    let a = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7);
    let b = Quat::from_axis_angle(Vec3::Y, 1.1);
    let v = Vec3::new(1.0, 2.0, 3.0);

    let ga = glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    let gb = glam::Quat::from_axis_angle(glam::Vec3::Y, 1.1);
    let gv = glam::Vec3::new(1.0, 2.0, 3.0);

    let mut g = c.benchmark_group("quat_mul_vec3");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a).mul_vec3(black_box(v)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga).mul_vec3(black_box(gv)))
    });
    g.finish();

    let mut g = c.benchmark_group("quat_mul_quat");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a) * black_box(b))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga) * black_box(gb))
    });
    g.finish();

    let mut g = c.benchmark_group("quat_slerp");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| black_box(a).slerp(black_box(b), black_box(0.35)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| black_box(ga).slerp(black_box(gb), black_box(0.35)))
    });
    g.finish();

    let mut g = c.benchmark_group("mat4_from_quat");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| Mat4::from_quat(black_box(a)))
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| glam::Mat4::from_quat(black_box(ga)))
    });
    g.finish();
}

/// Loops over arrays: this is how the library is used in practice, and lets
/// everything inline (no per-call `black_box` boundary like the single-op benches).
fn throughput(c: &mut Criterion) {
    const N: usize = 4096;

    let pts: Vec<Vec3> = (0..N)
        .map(|i| Vec3::new(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3))
        .collect();
    let gpts: Vec<glam::Vec3> = (0..N)
        .map(|i| glam::Vec3::new(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3))
        .collect();

    let m = Mat4::from_scale_rotation_translation(
        Vec3::new(1.5, 2.0, 0.5),
        Quat::from_axis_angle(Vec3::Y, 0.3),
        Vec3::new(4.0, 5.0, 6.0),
    );
    let gm = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(1.5, 2.0, 0.5),
        glam::Quat::from_axis_angle(glam::Vec3::Y, 0.3),
        glam::Vec3::new(4.0, 5.0, 6.0),
    );

    let mut g = c.benchmark_group("transform_4096_points");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| {
            let mut acc = Vec3::ZERO;
            for p in black_box(&pts) {
                acc += m.transform_point3(*p);
            }
            acc
        })
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| {
            let mut acc = glam::Vec3::ZERO;
            for p in black_box(&gpts) {
                acc += gm.transform_point3(*p);
            }
            acc
        })
    });
    g.finish();

    let mut g = c.benchmark_group("dot_4096");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| {
            let mut acc = 0.0f32;
            for w in black_box(&pts).windows(2) {
                acc += w[0].dot(w[1]);
            }
            acc
        })
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| {
            let mut acc = 0.0f32;
            for w in black_box(&gpts).windows(2) {
                acc += w[0].dot(w[1]);
            }
            acc
        })
    });
    g.finish();

    let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7);
    let gq = glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);

    let mut g = c.benchmark_group("quat_rotate_4096");
    g.bench_function("algebrix", |bench| {
        bench.iter(|| {
            let mut acc = Vec3::ZERO;
            for p in black_box(&pts) {
                acc += q.mul_vec3(*p);
            }
            acc
        })
    });
    g.bench_function("glam", |bench| {
        bench.iter(|| {
            let mut acc = glam::Vec3::ZERO;
            for p in black_box(&gpts) {
                acc += gq.mul_vec3(*p);
            }
            acc
        })
    });
    g.finish();
}

criterion_group!(benches, vec3_ops, vec4_ops, mat4_ops, quat_ops, throughput);
criterion_main!(benches);
