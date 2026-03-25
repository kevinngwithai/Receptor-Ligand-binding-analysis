import { SVD } from 'svd-js';

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export function getMatchingAtoms(atoms1: any[], atoms2: any[]) {
  // Group atoms by residue to handle numbering shifts
  const getResidues = (atoms: any[]) => {
    const residues: { resi: number, resn: string, atoms: any[] }[] = [];
    let currentResi = -99999;
    let currentRes: any = null;
    
    // Sort atoms by resi
    const sorted = [...atoms].sort((a, b) => a.resi - b.resi);
    
    for (const a of sorted) {
      if (a.resi !== currentResi) {
        currentRes = { resi: a.resi, resn: a.resn, atoms: [] };
        residues.push(currentRes);
        currentResi = a.resi;
      }
      currentRes.atoms.push(a);
    }
    return residues;
  };

  const res1 = getResidues(atoms1);
  const res2 = getResidues(atoms2);

  // Needleman-Wunsch sequence alignment
  const n = res1.length;
  const m = res2.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  const trace = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));

  for (let i = 1; i <= n; i++) {
    dp[i][0] = -i;
    trace[i][0] = 1; // up
  }
  for (let j = 1; j <= m; j++) {
    dp[0][j] = -j;
    trace[0][j] = 2; // left
  }

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const matchScore = res1[i - 1].resn === res2[j - 1].resn ? 2 : -1;
      const scoreDiag = dp[i - 1][j - 1] + matchScore;
      const scoreUp = dp[i - 1][j] - 1;
      const scoreLeft = dp[i][j - 1] - 1;

      if (scoreDiag >= scoreUp && scoreDiag >= scoreLeft) {
        dp[i][j] = scoreDiag;
        trace[i][j] = 0;
      } else if (scoreUp >= scoreLeft) {
        dp[i][j] = scoreUp;
        trace[i][j] = 1;
      } else {
        dp[i][j] = scoreLeft;
        trace[i][j] = 2;
      }
    }
  }

  const matched1: Point3D[] = [];
  const matched2: Point3D[] = [];

  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (trace[i][j] === 0) {
      const r1 = res1[i - 1];
      const r2 = res2[j - 1];
      
      const map2 = new Map();
      for (const a of r2.atoms) {
        map2.set(a.atom, a);
      }
      
      for (const a1 of r1.atoms) {
        if (map2.has(a1.atom)) {
          const a2 = map2.get(a1.atom);
          matched1.push({ x: a1.x, y: a1.y, z: a1.z });
          matched2.push({ x: a2.x, y: a2.y, z: a2.z });
        }
      }
      
      i--;
      j--;
    } else if (trace[i][j] === 1) {
      i--;
    } else {
      j--;
    }
  }

  return { matched1: matched1.reverse(), matched2: matched2.reverse() };
}

export function calculateCentroid(points: Point3D[]): Point3D {
  if (points.length === 0) return { x: 0, y: 0, z: 0 };
  let sumX = 0, sumY = 0, sumZ = 0;
  for (const p of points) {
    sumX += p.x;
    sumY += p.y;
    sumZ += p.z;
  }
  return {
    x: sumX / points.length,
    y: sumY / points.length,
    z: sumZ / points.length,
  };
}

export function kabsch(movingPoints: Point3D[], refPoints: Point3D[]) {
  if (movingPoints.length !== refPoints.length || movingPoints.length === 0) {
    throw new Error("Point sets must have the same non-zero length.");
  }

  const n = movingPoints.length;

  // 1. Calculate centroids
  const cMoving = calculateCentroid(movingPoints);
  const cRef = calculateCentroid(refPoints);

  // 2. Center the points
  const pCentered = movingPoints.map(p => [p.x - cMoving.x, p.y - cMoving.y, p.z - cMoving.z]);
  const qCentered = refPoints.map(p => [p.x - cRef.x, p.y - cRef.y, p.z - cRef.z]);

  // 3. Calculate covariance matrix H = P^T * Q
  // P is n x 3, Q is n x 3
  // H is 3 x 3
  const H = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];

  for (let i = 0; i < n; i++) {
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        H[row][col] += pCentered[i][row] * qCentered[i][col];
      }
    }
  }

  // 4. SVD of H
  // svd-js returns { u, v, q } where H = u * diag(q) * v^T
  const { u, v } = SVD(H, true, true, 1e-10, 1e-10);

  // 5. Calculate rotation matrix R = V * U^T
  // v is 3x3, u is 3x3
  let R = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];

  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      for (let k = 0; k < 3; k++) {
        R[row][col] += v[row][k] * u[col][k]; // u^T is u[col][k]
      }
    }
  }

  // 6. Check for reflection
  const detR = 
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);

  if (detR < 0) {
    // Multiply 3rd column of V by -1
    for (let row = 0; row < 3; row++) {
      v[row][2] *= -1;
    }
    // Recompute R
    R = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        for (let k = 0; k < 3; k++) {
          R[row][col] += v[row][k] * u[col][k];
        }
      }
    }
  }

  // 7. Calculate translation t = cRef - R * cMoving
  const t = {
    x: cRef.x - (R[0][0] * cMoving.x + R[0][1] * cMoving.y + R[0][2] * cMoving.z),
    y: cRef.y - (R[1][0] * cMoving.x + R[1][1] * cMoving.y + R[1][2] * cMoving.z),
    z: cRef.z - (R[2][0] * cMoving.x + R[2][1] * cMoving.y + R[2][2] * cMoving.z)
  };

  return { R, t };
}

export function applyTransform(points: Point3D[], R: number[][], t: Point3D): Point3D[] {
  return points.map(p => ({
    x: R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z + t.x,
    y: R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z + t.y,
    z: R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z + t.z
  }));
}
