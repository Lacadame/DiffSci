#!/usr/bin/env python
"""
Re-run drenagem + permeabilidade relativa reaproveitando redes SNOW2 já extraídas.

Propósito:
  Este script é uma variante "leve" do 0005b-porosity-field-new-metrics-evaluator.py
  que PULA a extração SNOW2 (a etapa mais cara) e reutiliza arquivos .network.npz
  salvos por uma rodada anterior. Isto permite explorar diferentes regimes físicos
  (contact angle, surface tension, face de invasão) com custo drasticamente menor.

  Use este script quando quiser variar:
    - contact angle (ex: 30 water-wet, 90 mixed, 120 oil-wet, 140 MICP)
    - surface tension (ex: 0.03 N/m óleo/água, 0.48 N/m mercúrio/ar)
    - inlet face (xmin/ymin/zmin) para medir anisotropia de fluxo bifásico

  Para recalcular do zero a partir de volumes .npy, use o 0005b original.

Fluxo por rede:
  1. Carrega .network.npz → dict compatível com PoreSpy
  2. PoreNetworkPermeability.from_porespy_network(...) — pula SNOW2
  3. calculate_absolute_permeability() — K em 3 direções
  4. run_drainage_simulation(inlet_face, contact_angle, surface_tension)
  5. calculate_relative_permeability_curves() — kr_w/kr_nw por direção
  6. Salva tudo em .npz (mesmo formato do 0005b)

Uso:
    python scripts/0005b-rerun-drainage.py \\
        --network-paths path/to/1024_0.network.npz,path/to/1024_1.network.npz \\
        --labels 1024_0,1024_1 \\
        --volume-lengths 1024,1024 \\
        --voxel-length 2.25e-6 \\
        --contact-angle 30 \\
        --surface-tension 0.03 \\
        --inlet-face xmin \\
        --output metrics_waterwet_xmin.npz
"""

import argparse
import os
import time

import numpy as np

from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability


def parse_args():
    parser = argparse.ArgumentParser(
        description='Re-drenagem reaproveitando redes SNOW2 pré-extraídas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--network-paths', type=str, required=True,
        help='Lista de caminhos .network.npz separados por vírgula'
    )
    parser.add_argument(
        '--labels', type=str, required=True,
        help='Lista de rótulos (um por rede) separados por vírgula. '
             'Usados como prefixo nas chaves do .npz de saída.'
    )
    parser.add_argument(
        '--volume-lengths', type=str, required=True,
        help='Lista de edge lengths (voxels) do cubo de onde cada rede foi extraída, '
             'separados por vírgula. Usado em A = L² e na Lei de Darcy.'
    )
    parser.add_argument(
        '--voxel-length', type=float, required=True,
        help='Tamanho do voxel em metros (ex: 2.25e-6 para Berea eleven_sandstones)'
    )
    parser.add_argument(
        '--contact-angle', type=float, default=140.0,
        help='Contact angle em graus (default: 140, regime MICP)'
    )
    parser.add_argument(
        '--surface-tension', type=float, default=0.48,
        help='Surface tension em N/m (default: 0.48, mercúrio/ar)'
    )
    parser.add_argument(
        '--inlet-face', type=str, default='xmin',
        choices=['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'],
        help='Face pela qual a fase não-molhante invade (default: xmin)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Caminho do .npz de saída'
    )
    parser.add_argument(
        '--regime-name', type=str, default='custom',
        help='Nome do regime (ex: micp, oilwet, waterwet, mixedwet). '
             'Apenas metadata — não afeta cálculo.'
    )
    return parser.parse_args()


def load_porespy_network(path):
    """Carrega .network.npz e converte para dict compatível com openpnm.io.network_from_porespy."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def compute_metrics(network_path, volume_length, voxel_size,
                    contact_angle, surface_tension, inlet_face):
    """Retorna dict com porosity, K_abs (x/y/z/mean), Sw, Pc, kr_w, kr_nw."""
    network_dict = load_porespy_network(network_path)

    pn_wrapper = PoreNetworkPermeability.from_porespy_network(
        network_dict,
        volume_length=volume_length,
        voxel_size=voxel_size,
    )

    # ─── OVERRIDE DE COMPATIBILIDADE COM BASELINE E2 ────────────────────────
    # O baseline do experimento 20260316 (eval, bl, network_analysis,
    # diagnostics) foi calculado quando _setup_network_geometry usava
    # throat.inscribed_diameter. O código foi alterado para usar
    # throat.equivalent_diameter (argumento: físicamente mais correto para
    # Hagen-Poiseuille em seções não-circulares). Essa mudança produz K
    # ~3.8× maior do que o baseline.
    #
    # Para manter compatibilidade com os .npz do E2 já gerados, forçamos
    # throat.diameter = inscribed_diameter aqui. A decisão científica
    # "inscribed vs equivalent" fica como ponto pendente para uma rodada
    # futura.
    # ────────────────────────────────────────────────────────────────────────
    pn_wrapper.pn['throat.diameter'] = pn_wrapper.pn['throat.inscribed_diameter']
    pn_wrapper.pn['throat.radius'] = pn_wrapper.pn['throat.diameter'] / 2.0

    abs_perm = pn_wrapper.calculate_absolute_permeability()
    _ = pn_wrapper.run_drainage_simulation(
        inlet_face=inlet_face,
        contact_angle=contact_angle,
        surface_tension=surface_tension,
    )
    rel_perm = pn_wrapper.calculate_relative_permeability_curves()

    return {
        'K_abs_x': abs_perm.K_x,
        'K_abs_y': abs_perm.K_y,
        'K_abs_z': abs_perm.K_z,
        'K_abs_mean': abs_perm.K_mean,
        'K_abs_x_physical': abs_perm.K_x_physical,
        'K_abs_y_physical': abs_perm.K_y_physical,
        'K_abs_z_physical': abs_perm.K_z_physical,
        'K_abs_mean_physical': abs_perm.K_mean_physical,
        'Sw': rel_perm.Sw,
        'Snw': rel_perm.Snwp,
        'Pc': rel_perm.Pc,
        'kr_wetting': rel_perm.kr_wetting,
        'kr_nonwetting': rel_perm.kr_nonwetting,
        'kr_wetting_mean': rel_perm.kr_wetting_mean,
        'kr_nonwetting_mean': rel_perm.kr_nonwetting_mean,
    }


def main():
    args = parse_args()

    network_paths = [p.strip() for p in args.network_paths.split(',')]
    labels = [l.strip() for l in args.labels.split(',')]
    volume_lengths = [int(x.strip()) for x in args.volume_lengths.split(',')]

    if not (len(network_paths) == len(labels) == len(volume_lengths)):
        raise ValueError(
            f"--network-paths ({len(network_paths)}), --labels ({len(labels)}) "
            f"e --volume-lengths ({len(volume_lengths)}) devem ter o mesmo tamanho."
        )

    print("=" * 70)
    print(f"Re-drenagem multi-regime — regime: {args.regime_name}")
    print("=" * 70)
    print(f"  Networks:        {len(network_paths)}")
    print(f"  Contact angle:   {args.contact_angle}°")
    print(f"  Surface tension: {args.surface_tension} N/m")
    print(f"  Inlet face:      {args.inlet_face}")
    print(f"  Voxel length:    {args.voxel_length:.3e} m")
    print(f"  Output:          {args.output}")
    print("=" * 70)
    print()

    results = {
        'regime_name': args.regime_name,
        'contact_angle': args.contact_angle,
        'surface_tension': args.surface_tension,
        'inlet_face': args.inlet_face,
        'voxel_length': args.voxel_length,
        'labels': np.array(labels),
        'network_paths': np.array(network_paths),
        'volume_lengths': np.array(volume_lengths),
    }
    timing = {}

    for i, (path, label, L) in enumerate(zip(network_paths, labels, volume_lengths)):
        print(f"[{i+1}/{len(network_paths)}] {label} (L={L}, {os.path.basename(path)})")
        t0 = time.time()
        try:
            m = compute_metrics(
                network_path=path,
                volume_length=L,
                voxel_size=args.voxel_length,
                contact_angle=args.contact_angle,
                surface_tension=args.surface_tension,
                inlet_face=args.inlet_face,
            )
        except Exception as e:
            print(f"    Erro: {e}")
            continue
        dt = time.time() - t0
        timing[label] = dt
        for key, value in m.items():
            results[f'{label}_{key}'] = value
        print(f"    K_abs_mean: {m['K_abs_mean_physical']*1e15:.2f} ×10⁻¹⁵ m² | "
              f"{len(m['Sw'])} pontos Sw | {dt:.1f}s")

    results['timing_labels'] = np.array(list(timing.keys()))
    results['timing_values'] = np.array(list(timing.values()))
    results['total_time'] = sum(timing.values())

    # Salvar
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Arrays de comprimentos variáveis (Sw, Pc, kr) viram object arrays
    save_dict = {}
    for k, v in results.items():
        save_dict[k] = v

    np.savez(args.output, **save_dict)
    print()
    print(f"=== Concluído em {results['total_time']:.1f}s ===")
    print(f"Saída: {args.output}")


if __name__ == '__main__':
    main()
