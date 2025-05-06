import math
from enum import IntEnum, unique

import lhapdf
import numpy as np
import scipy
import vegas


# fmt: off

@unique
class Order(IntEnum):
    """Parton Id's"""

    LO = 0
    NLO = 1
    NLO_ONLY = 2

    def __str__(self):
        return self.name.lower()


@unique
class PID(IntEnum):
    """Parton Id's"""

    G = 0  # gluon
    D = 1  # down-type
    U = 2  # up-type

    def __str__(self):
        return self.name.lower()


class Parameters(object):
    """very simple class to manage Standard Model Parameters"""

    # > conversion factor from GeV^{-2} into picobarns [pb]
    GeVpb = 0.3893793656e9

    def __init__(self, **kwargs):
        # > these are the independent variables we chose:
        # >  *  sw2 = sin^2(theta_w) with the weak mixing angle theta_w
        # >  *  (MZ, GZ) = mass & width of Z-boson
        self.sw2  = kwargs.pop("sw2", 0.223013225326783359748)
        self.MZ   = kwargs.pop("MZ", 91.1876)
        self.GZ   = kwargs.pop("GZ", 2.4952)
        self.sPDF = kwargs.pop("sPDF", "NNPDF31_nnlo_as_0118_luxqed")
        self.iPDF = kwargs.pop("iPDF", 0)
        if len(kwargs) > 0:
            raise RuntimeError("passed unknown parameter(s): {}".format(kwargs))

        # > we'll cache the PDF set for performance
        lhapdf.setVerbosity(0)
        self.pdf = lhapdf.mkPDF(self.sPDF, self.iPDF)

        # > let's store some more constants (l, u, d = lepton, up-quark, down-quark)
        self.Ql = -1.0;        self.I3l = -1.0 / 2.0  # charge & weak isospin
        self.Qu = +2.0 / 3.0;  self.I3u = +1.0 / 2.0
        self.Qd = -1.0 / 3.0;  self.I3d = -1.0 / 2.0
        self.alpha = 1.0 / 132.184179137044878871  # G_mu scheme
        self.Nc = 3  # number of colours

        # > and some derived quantities
        self.sw = math.sqrt(self.sw2)
        self.cw2 = 1.0 - self.sw2  # cos^2 = 1-sin^2
        self.cw = math.sqrt(self.cw2)

    # > vector & axial-vector couplings to Z-boson
    @property
    def vl(self) -> float:
        return (self.I3l - 2 * self.Ql * self.sw2) / (2.0 * self.sw * self.cw)
    @property
    def al(self) -> float:
        return self.I3l / (2.0 * self.sw * self.cw)
    def vq(self, qid: PID) -> float:
        if qid == PID.D:
            return (self.I3d - 2 * self.Qd * self.sw2) / (2.0 * self.sw * self.cw)
        if qid == PID.U:
            return (self.I3u - 2 * self.Qu * self.sw2) / (2.0 * self.sw * self.cw)
        raise RuntimeError("vq called with invalid qid: {}".format(qid))
    def aq(self, qid: PID) -> float:
        if qid == PID.D:
            return self.I3d / (2.0 * self.sw * self.cw)
        if qid == PID.U:
            return self.I3u / (2.0 * self.sw * self.cw)
        raise RuntimeError("aq called with invalid qid: {}".format(qid))
    def Qq(self, qid: PID) -> float:
        if qid == PID.D:
            return self.Qd
        if qid == PID.U:
            return self.Qu
        raise RuntimeError("Qq called with invalid qid: {}".format(qid))

    # > colour factors
    @property
    def CA(self) -> float:
        return self.Nc
    @property
    def CF(self) -> float:
        return (self.Nc**2 - 1.0) / (2.0 * self.Nc)

    # > average factors (spin & colour)
    def navg(self, pid: PID) -> int:
        if pid == PID.G:
            return 2 * (self.Nc**2 - 1)
        if (pid == PID.D) or (pid == PID.U):
            return 2 * self.Nc
        raise RuntimeError("navg called with invalid pid: {}".format(pid))

    # > the Z-boson propagator
    def propZ(self, s: float) -> complex:
        return s / (s - complex(self.MZ**2, self.GZ * self.MZ))

# > we immediately instantiate an object (default values) in global scope
PARAM = Parameters()


# > Lepton and Hadron (tree-level) form factors
def L_yy(Q2: float, p=PARAM) -> float:
    return 2 / 3.0 * p.alpha / Q2 * p.Ql**2
def L_ZZ(Q2: float, p=PARAM) -> float:
    return 2 / 3.0 * p.alpha / Q2 * (p.vl**2 + p.al**2) * abs(p.propZ(Q2)) ** 2
def L_Zy(Q2: float, p=PARAM) -> float:
    return 2 / 3.0 * p.alpha / Q2 * p.vl * p.Ql * p.propZ(Q2).real

def H0_yy(Q2: float, qid: PID, p=PARAM) -> float:
    return 16 * math.pi * p.Nc * p.alpha * Q2 * p.Qq(qid) ** 2
def H0_ZZ(Q2: float, qid: PID, p=PARAM) -> float:
    return 16 * math.pi * p.Nc * p.alpha * Q2 * (p.vq(qid) ** 2 + p.aq(qid) ** 2)
def H0_Zy(Q2: float, qid: PID, p=PARAM) -> float:
    return 16 * math.pi * p.Nc * p.alpha * Q2 * p.vq(qid) * p.Qq(qid)

def H0xL(Q2: float, qid: PID, par=PARAM) -> float:
    return (
              L_yy(Q2, par) * H0_yy(Q2, qid, par)
        +     L_ZZ(Q2, par) * H0_ZZ(Q2, qid, par)
        + 2 * L_Zy(Q2, par) * H0_Zy(Q2, qid, par)
    )


# > Parton lulminosities including averaging factors for initial-state d.o.f.
def lumi(ida: PID, idb: PID, xa: float, xb: float, muF: float, p=PARAM) -> float:
    if (ida, idb) == (PID.D, PID.D):
        return (
              p.pdf.xfxQ(+1, xa, muF) * p.pdf.xfxQ(-1, xb, muF)  # (d,dbar)
            + p.pdf.xfxQ(+3, xa, muF) * p.pdf.xfxQ(-3, xb, muF)  # (s,sbar)
            + p.pdf.xfxQ(+5, xa, muF) * p.pdf.xfxQ(-5, xb, muF)  # (b,bbar)
            + p.pdf.xfxQ(-1, xa, muF) * p.pdf.xfxQ(+1, xb, muF)  # (dbar,d)
            + p.pdf.xfxQ(-3, xa, muF) * p.pdf.xfxQ(+3, xb, muF)  # (sbar,s)
            + p.pdf.xfxQ(-5, xa, muF) * p.pdf.xfxQ(+5, xb, muF)  # (bbar,b)
        ) / (xa * xb * p.navg(ida) * p.navg(idb))
    if (ida, idb) == (PID.U, PID.U):
        return (
              p.pdf.xfxQ(+2, xa, muF) * p.pdf.xfxQ(-2, xb, muF)  # (u,ubar)
            + p.pdf.xfxQ(+4, xa, muF) * p.pdf.xfxQ(-4, xb, muF)  # (c,cbar)
            + p.pdf.xfxQ(-2, xa, muF) * p.pdf.xfxQ(+2, xb, muF)  # (ubar,u)
            + p.pdf.xfxQ(-4, xa, muF) * p.pdf.xfxQ(+4, xb, muF)  # (cbar,c)
        ) / (xa * xb * p.navg(ida) * p.navg(idb))
    if (ida, idb) == (PID.D, PID.G):
        return (
              p.pdf.xfxQ(+1, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (d,g)
            + p.pdf.xfxQ(+3, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (s,g)
            + p.pdf.xfxQ(+5, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (b,g)
            + p.pdf.xfxQ(-1, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (dbar,g)
            + p.pdf.xfxQ(-3, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (sbar,g)
            + p.pdf.xfxQ(-5, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (bbar,g)
        ) / (xa * xb * p.navg(ida) * p.navg(idb))
    if (ida, idb) == (PID.G, PID.D):
        return (
              p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(+1, xb, muF)  # (g,d)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(+3, xb, muF)  # (g,s)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(+5, xb, muF)  # (g,b)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(-1, xb, muF)  # (g,dbar)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(-3, xb, muF)  # (g,sbar)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(-5, xb, muF)  # (g,bbar)
        ) / (xa * xb * p.navg(ida) * p.navg(idb))
    if (ida, idb) == (PID.U, PID.G):
        return (
              p.pdf.xfxQ(+2, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (u,g)
            + p.pdf.xfxQ(+4, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (c,g)
            + p.pdf.xfxQ(-2, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (ubar,g)
            + p.pdf.xfxQ(-4, xa, muF) * p.pdf.xfxQ(0, xb, muF)  # (cbar,g)
        ) / (xa * xb * p.navg(ida) * p.navg(idb))
    if (ida, idb) == (PID.G, PID.U):
        return (
              p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(+2, xb, muF)  # (g,u)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(+4, xb, muF)  # (g,c)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(-2, xb, muF)  # (g,ubar)
            + p.pdf.xfxQ(0, xa, muF) * p.pdf.xfxQ(-4, xb, muF)  # (g,cbar)
        ) / (xa * xb * p.navg(ida) * p.navg(idb))
    raise RuntimeError("lumi called with invalid ids: ({},{})".format(ida, idb))


def integrand(
    order: Order,
    za: float, zb: float,
    xa: float, xb: float,
    Q: float,
    facR: float = 1.0, facF: float = 1.0,
    maxQT: float = float("inf"),
    p=PARAM,
):
    res = 0.0

    # > the Born MEs
    cross_dn = H0xL(Q**2, PID.D, p)
    cross_up = H0xL(Q**2, PID.U, p)

    # > some convenient abbreviations to cache
    oMxa = 1 - xa;  oMza = 1 - za;  oPza = 1 + za
    oMxb = 1 - xb;  oMzb = 1 - zb;  oPzb = 1 + zb
    lnoMxa = math.log(oMxa);  lnoMza = math.log(oMza);  lnoPza = math.log(oPza)
    lnoMxb = math.log(oMxb);  lnoMzb = math.log(oMzb);  lnoPzb = math.log(oPzb)
    lnF = -2.0 * math.log(facF);  ln2 = math.log(2.0)
    D0xa = 1.0 / oMxa;  D1xa = lnoMxa / oMxa;  D2xa = lnoMxa**2 / oMxa
    D0za = 1.0 / oMza;  D1za = lnoMza / oMza
    D0xb = 1.0 / oMxb;  D1xb = lnoMxb / oMxb;  D2xb = lnoMxb**2 / oMxb
    D0zb = 1.0 / oMzb;  D1zb = lnoMzb / oMzb

    # > order prefactors
    fac_LO: float = 1.0
    fac_NLO: float = (p.pdf.alphasQ(facR * Q) / math.pi) * p.CF
    # > reset depending on the chosen order
    if order == Order.LO:
        fac_NLO = 0.0
    if order == Order.NLO_ONLY:
        fac_LO = 0.0

    # > QT: cut only acts on the RaRb region (any delta => QT == 0)
    QT: float = Q * math.sqrt(oMza * oPza * oMzb * oPzb) / (za + zb)
    fac_maxQT = 1.0 if QT < maxQT else 0.0

    ### the corrections decomposed into "regions"

    # > LO: only qq channel
    LO_EaEb_qq = D0xa * D0xb
    # > NLO: qq channel
    EaEb_qq =  1 / 2.0 * D0xa * D0xb * (math.pi**2 - 8)
    EaEb_qq += D0za * D0zb * (1)
    EaEb_qq += D0za * D1xb * (-1)
    EaEb_qq += D0zb * D1xa * (-1)
    EaEb_qq += D1xa * D1xb * (1)
    EaEb_qq += D1za * (-(D0xb))
    EaEb_qq += D1zb * (-(D0xa))
    EaEb_qq += D2xa * (1 / 2.0 * D0xb)
    EaEb_qq += D2xb * (1 / 2.0 * D0xa)
    EaEb_qq += lnF * (3 / 2.0 * D0xa * D0xb)
    EaEb_qq += lnF * D0za * (-D0xb)
    EaEb_qq += lnF * D0zb * (-D0xa)
    EaEb_qq += lnF * D1xa * (D0xb)
    EaEb_qq += lnF * D1xb * (D0xa)
    EaRb_qq =  1 / 2.0 * oMzb * D0xa * (ln2 + lnoMzb - lnoPzb + 1)
    EaRb_qq += D0za * (-1 / 2.0 * oMzb)
    EaRb_qq += D0za * D0zb * (-zb)
    EaRb_qq += D0zb * (zb * D0xa * (ln2 - lnoPzb))
    EaRb_qq += D0zb * D1xa * (zb)
    EaRb_qq += D1xa * (1 / 2.0 * oMzb)
    EaRb_qq += D1zb * (zb * D0xa)
    EaRb_qq += lnF * (1 / 2.0 * oMzb * D0xa)
    EaRb_qq += lnF * D0zb * (zb * D0xa)
    RaEb_qq =  1 / 2.0 * oMza * D0xb * (ln2 + lnoMza - lnoPza + 1)
    RaEb_qq += D0za * (za * D0xb * (ln2 - lnoPza))
    RaEb_qq += D0za * D0zb * (-za)
    RaEb_qq += D0za * D1xb * (za)
    RaEb_qq += D0zb * (-1 / 2.0 * oMza)
    RaEb_qq += D1xb * (1 / 2.0 * oMza)
    RaEb_qq += D1za * (za * D0xb)
    RaEb_qq += lnF * (1 / 2.0 * oMza * D0xb)
    RaEb_qq += lnF * D0za * (za * D0xb)
    RaRb_qq =  D0za * (za**2 * oMzb * oPza**-1 * oPzb * (za + zb) ** -2 * (za * zb + 1))
    RaRb_qq += D0za * D0zb * (2 * za * zb * oPza**-1 * oPzb**-1 * (za * zb + 1))
    RaRb_qq += D0zb * (zb**2 * oMza * oPza * oPzb**-1 * (za + zb) ** -2 * (za * zb + 1))
    # > NLO: qg channel
    EaRb_qg =  1 / 2.0 * D0xa * (2 * zb - 2 * zb**2 + ln2 + lnoMzb - lnoPzb - 2 * zb * ln2 - 2 * zb * lnoMzb + 2 * zb * lnoPzb + 2 * zb**2 * ln2 + 2 * zb**2 * lnoMzb - 2 * zb**2 * lnoPzb)
    EaRb_qg += D0za * (-1 / 2.0 * (-2 * zb + 2 * zb**2 + 1))
    EaRb_qg += D1xa * (1 / 2.0 * (-2 * zb + 2 * zb**2 + 1))
    EaRb_qg += lnF * 1 / 2.0 * D0xa * (-2 * zb + 2 * zb**2 + 1)
    RaRb_qg =  za * zb**2 * oMza * oPza * (za + zb) ** -3 * (za * zb + 1)
    RaRb_qg += D0za * (za * oPza**-1 * (za + zb) ** -2 * (za * zb + 1) * (za + zb - 2 * za**2 * zb + 2 * za**2 * zb**3))
    # > NLO: gq channel  (the above with a <-> b & region swaps)
    RaEb_gq =  1 / 2.0 * D0xb * (2 * za - 2 * za**2 + ln2 + lnoMza - lnoPza - 2 * za * ln2 - 2 * za * lnoMza + 2 * za * lnoPza + 2 * za**2 * ln2 + 2 * za**2 * lnoMza - 2 * za**2 * lnoPza)
    RaEb_gq += D0zb * (-1 / 2.0 * (-2 * za + 2 * za**2 + 1))
    RaEb_gq += D1xb * (1 / 2.0 * (-2 * za + 2 * za**2 + 1))
    RaEb_gq += lnF * 1 / 2.0 * D0xb * (-2 * za + 2 * za**2 + 1)
    RaRb_gq =  zb * za**2 * oMzb * oPzb * (zb + za) ** -3 * (zb * za + 1)
    RaRb_gq += D0zb * (zb * oPzb**-1 * (zb + za) ** -2 * (zb * za + 1) * (zb + za - 2 * zb**2 * za + 2 * zb**2 * za**3))

    ### assembly of the "regions"

    # > EaEb
    res += cross_dn * (fac_LO * LO_EaEb_qq + fac_NLO * EaEb_qq) * lumi(PID.D, PID.D, xa, xb, facF * Q, p)
    res += cross_up * (fac_LO * LO_EaEb_qq + fac_NLO * EaEb_qq) * lumi(PID.U, PID.U, xa, xb, facF * Q, p)
    # > EaRb
    res += cross_dn * fac_NLO * (
          EaRb_qq * lumi(PID.D, PID.D, xa, xb / zb, facF * Q, p)
        + EaRb_qg * lumi(PID.D, PID.G, xa, xb / zb, facF * Q, p)
    ) / zb
    res += cross_up * fac_NLO * (
          EaRb_qq * lumi(PID.U, PID.U, xa, xb / zb, facF * Q, p)
        + EaRb_qg * lumi(PID.U, PID.G, xa, xb / zb, facF * Q, p)
    ) / zb
    # > RaEb
    res += cross_dn * fac_NLO * (
          RaEb_qq * lumi(PID.D, PID.D, xa / za, xb, facF * Q, p)
        + RaEb_gq * lumi(PID.G, PID.D, xa / za, xb, facF * Q, p)
    ) / za
    res += cross_up * fac_NLO * (
          RaEb_qq * lumi(PID.U, PID.U, xa / za, xb, facF * Q, p)
        + RaEb_gq * lumi(PID.G, PID.U, xa / za, xb, facF * Q, p)
    ) / za
    # > RaRb  (only region where the maxQT constraint can act on)
    res += cross_dn * fac_NLO * fac_maxQT * (
          RaRb_qq * lumi(PID.D, PID.D, xa / za, xb / zb, facF * Q, p)
        + RaRb_qg * lumi(PID.D, PID.G, xa / za, xb / zb, facF * Q, p)
        + RaRb_gq * lumi(PID.G, PID.D, xa / za, xb / zb, facF * Q, p)
    ) / (za * zb)
    res += cross_up * fac_NLO * fac_maxQT * (
          RaRb_qq * lumi(PID.U, PID.U, xa / za, xb / zb, facF * Q, p)
        + RaRb_qg * lumi(PID.U, PID.G, xa / za, xb / zb, facF * Q, p)
        + RaRb_gq * lumi(PID.G, PID.U, xa / za, xb / zb, facF * Q, p)
    ) / (za * zb)

    return res


def diff_cross_NLO(
    Ecm: float, Mll: float, Yll: float,
    facR: float = 1.0, facF: float = 1.0,
    maxQT: float = float("inf"),
    p=PARAM,
) -> float:
    xa = (Mll / Ecm) * math.exp(+Yll)
    xb = (Mll / Ecm) * math.exp(-Yll)
    if xa > 1 or xb > 1:
        return 0.0
    conv = scipy.integrate.nquad(lambda za, zb: integrand(Order.NLO, za, zb, xa, xb, Mll, facR, facF, maxQT, p), [[xa, 1], [xb, 1]])
    return p.GeVpb * conv[0] * Mll / Ecm**4 / (xa * xb)


def full_integrand(
    order: Order,
    Ecm: float, Mll: float, Yll: float,
    ra: float, rb: float,
    facR: float = 1.0, facF: float = 1.0,
    maxQT: float = float("inf"),
    p=PARAM,
) -> float:
    xa = (Mll / Ecm) * math.exp(+Yll)
    xb = (Mll / Ecm) * math.exp(-Yll)
    if xa > 1 or xb > 1:
        return 0.0
    za = xa + ra * (1 - xa)
    zb = xb + rb * (1 - xb)
    jac = (1 - xa) * (1 - xb)
    return p.GeVpb * jac * integrand(order, za, zb, xa, xb, Mll, facR, facF, maxQT, p) * Mll / Ecm**4 / (xa * xb)


def full_integrands(
    order: Order,
    Ecm: float, Mll: float, Yll: float,
    ra: float, rb: float,
    facsR: list[float] = [1.0], facsF: list[float] = [1.0],
    maxQT: float = float("inf"),
    p=PARAM,
) -> list[float]:
    xa = (Mll / Ecm) * math.exp(+Yll)
    xb = (Mll / Ecm) * math.exp(-Yll)
    if xa > 1 or xb > 1:
        return [0.0 for facR in facsR for facF in facsF]
    za = xa + ra * (1 - xa)
    zb = xb + rb * (1 - xb)
    jac = (1 - xa) * (1 - xb)
    return [ p.GeVpb * jac * integrand(order, za, zb, xa, xb, Mll, facR, facF, maxQT, p) * Mll / Ecm**4 / (xa * xb) for facR in facsR for facF in facsF]

def parse_scl_var(res):
    res_ctr = res[0].mean
    res_err = res[0].sdev
    res_scl = [res.mean for res in res]
    res_dup = max(res_scl) - res_ctr
    res_ddn = min(res_scl) - res_ctr
    return res_ctr, res_err, res_dup, res_ddn

def main():
    # > com energy of the hadron collider
    Ecm = 8000.0

    # # > total cross section (fiducial via "cuts" on Mll & Yll as integration ranges)
    # tot_integ = vegas.Integrator([[80.0, 100.0], [-3.6, +3.6], [0.0, 1.0], [0.0, 1.0]])
    # _       = tot_integ(lambda x: full_integrand(Order.LO,  Ecm, x[0], x[1], x[2], x[3], 1, 1), nitn=10, neval=500)
    # tot_LO  = tot_integ(lambda x: full_integrand(Order.LO,  Ecm, x[0], x[1], x[2], x[3], 1, 1), nitn=10, neval=5000)
    # _       = tot_integ(lambda x: full_integrand(Order.NLO, Ecm, x[0], x[1], x[2], x[3], 1, 1), nitn=10, neval=500)
    # tot_NLO = tot_integ(lambda x: full_integrand(Order.NLO, Ecm, x[0], x[1], x[2], x[3], 1, 1), nitn=10, neval=5000)
    # # print(tot_NLO.summary())
    # print("\n# total cross section in pb units:  LO +/- LO_err,  NLO +/- NLO_err")
    # print("#tot  {:e} {:e}  {:e} {:e}".format(tot_LO.mean, tot_LO.sdev, tot_NLO.mean, tot_NLO.sdev))

    # > total cross section (fiducial via "cuts" on Mll & Yll as integration ranges)
    tot_integ = vegas.Integrator([[80.0, 100.0], [-3.6, +3.6], [0.0, 1.0], [0.0, 1.0]])
    _       = tot_integ(lambda x: full_integrands(Order.LO,  Ecm, x[0], x[1], x[2], x[3], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=300)
    tot_LO  = tot_integ(lambda x: full_integrands(Order.LO,  Ecm, x[0], x[1], x[2], x[3], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=2000)

    _       = tot_integ(lambda x: full_integrands(Order.NLO, Ecm, x[0], x[1], x[2], x[3], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=300)
    tot_NLO = tot_integ(lambda x: full_integrands(Order.NLO, Ecm, x[0], x[1], x[2], x[3], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=2000)
    LO_res, LO_err, LO_dup, LO_ddn = parse_scl_var(tot_LO)
    NLO_res, NLO_err, NLO_dup, NLO_ddn = parse_scl_var(tot_NLO)
    print("\n# total cross section in pb units:  LO LO_err +LO_up -LO_dn,  NLO NLO_err +NLO_up -NLO_dn")
    print("#tot  {:e} {:e} {:e} {:e}  {:e} {:e} {:e} {:e}".format(LO_res, LO_err, LO_dup, LO_ddn, NLO_res, NLO_err, NLO_dup, NLO_ddn))


    # # > rapidity distribution
    # Y_integ = vegas.Integrator([[80.0, 100.0], [0.0, 1.0], [0.0, 1.0]])
    # print("\n# rapidity distribution:  Y,  LO +/- LO_err,  NLO +/- NLO_err")
    # # for Yll in np.linspace(-3.6, 3.6, 73):
    # for Yll in np.linspace(0.0, 3.6, 37):
    #     _     = Y_integ(lambda x: full_integrand(Order.LO,  Ecm, x[0], Yll, x[1], x[2]), nitn=10, neval=200)
    #     Y_LO  = Y_integ(lambda x: full_integrand(Order.LO,  Ecm, x[0], Yll, x[1], x[2]), nitn=10, neval=1000)
    #     _     = Y_integ(lambda x: full_integrand(Order.NLO, Ecm, x[0], Yll, x[1], x[2]), nitn=10, neval=200)
    #     Y_NLO = Y_integ(lambda x: full_integrand(Order.NLO, Ecm, x[0], Yll, x[1], x[2]), nitn=10, neval=1000)
    #     # print("#Yll  {:e}  {:e} {:e}  {:e} {:e}".format(Yll, Y_LO.mean, Y_LO.sdev, Y_NLO.mean, Y_NLO.sdev))
    #     print("#Yll  {:e}  {:e} {:e}  {:e} {:e}".format(Yll, 2*Y_LO.mean, 2*Y_LO.sdev, 2*Y_NLO.mean, 2*Y_NLO.sdev))

    # > rapidity distribution
    Y_integ = vegas.Integrator([[80.0, 100.0], [0.0, 1.0], [0.0, 1.0]])
    print("\n# rapidity distribution:  Y,  LO LO_err +LO_up -LO_dn,  NLO NLO_err +NLO_up -NLO_dn")
    # for Yll in np.linspace(-3.6, 3.6, 73):
    for Yll in np.linspace(0.0, 3.6, 37):
        _     = Y_integ(lambda x: full_integrands(Order.LO,  Ecm, x[0], Yll, x[1], x[2], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=300)
        Y_LO  = Y_integ(lambda x: full_integrands(Order.LO,  Ecm, x[0], Yll, x[1], x[2], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=2000)
        _     = Y_integ(lambda x: full_integrands(Order.NLO, Ecm, x[0], Yll, x[1], x[2], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=300)
        Y_NLO = Y_integ(lambda x: full_integrands(Order.NLO, Ecm, x[0], Yll, x[1], x[2], [1, 2, 0.5], [1, 2, 0.5]), nitn=10, neval=2000)
        LO_res, LO_err, LO_dup, LO_ddn = parse_scl_var(Y_LO)
        NLO_res, NLO_err, NLO_dup, NLO_ddn = parse_scl_var(Y_NLO)
        print("#Yll  {:e}  {:e} {:e} {:e} {:e}  {:e} {:e} {:e} {:e}".format(Yll,  2*LO_res, 2*LO_err, 2*LO_dup, 2*LO_ddn,  2*NLO_res, 2*NLO_err, 2*NLO_dup, 2*NLO_ddn))


    return

    # # > invariant mass distribution
    # M_integ = vegas.Integrator([[-3.6, +3.6], [0.0, 1.0], [0.0, 1.0]])
    # print("\n# invariant mass distribution:  M,  LO +/- LO_err,  NLO +/- NLO_err")
    # for Mll in np.linspace(10, 500, 100):
    #     _     = M_integ(lambda x: full_integrand(Order.LO,  Ecm, Mll, x[0], x[1], x[2]), nitn=10, neval=200)
    #     M_LO  = M_integ(lambda x: full_integrand(Order.LO,  Ecm, Mll, x[0], x[1], x[2]), nitn=10, neval=1000)
    #     _     = M_integ(lambda x: full_integrand(Order.NLO, Ecm, Mll, x[0], x[1], x[2]), nitn=10, neval=200)
    #     M_NLO = M_integ(lambda x: full_integrand(Order.NLO, Ecm, Mll, x[0], x[1], x[2]), nitn=10, neval=1000)
    #     print("#Mll  {:e}  {:e} {:e}  {:e} {:e}".format(Mll, M_LO.mean, M_LO.sdev, M_NLO.mean, M_NLO.sdev))

    # # > cumulant distribution in maxQT
    # QT_integ = vegas.Integrator([[80.0, 100.0], [-3.6, +3.6], [0.0, 1.0], [0.0, 1.0]])
    # print("\n# cumulant distribution in QT:  QT,  NLO +/- NLO_err,  DL")
    # for QT in np.logspace(-5, 3, 81):
    #     _      = QT_integ(lambda x: full_integrand(Order.NLO, Ecm, x[0], x[1], x[2], x[3], 1, 1, QT), nitn=10, neval=500)
    #     QT_NLO = QT_integ(lambda x: full_integrand(Order.NLO, Ecm, x[0], x[1], x[2], x[3], 1, 1, QT), nitn=10, neval=2000)
    #     QT_DL = tot_LO.mean * (1.0 - PARAM.pdf.alphasQ(PARAM.MZ) / math.pi * 2.0 * PARAM.CF * math.log(PARAM.MZ / QT) ** 2)
    #     print("#QT  {:e}  {:e} {:e}  {:e}".format(QT, QT_NLO.mean, QT_NLO.sdev, QT_DL))


# fmt: on


if __name__ == "__main__":
    main()
