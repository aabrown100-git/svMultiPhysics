
// The functions defined here replicate the Fortran functions defined in READMSH.f.

#include "Array.h"
#include "ComMod.h"
#include "Vector.h"

#include "all_fun.h"
#include "consts.h"
#include "load_msh.h"
#include "read_msh.h"
#include "utils.h"
#include "vtk_xml.h"
#include "vtk_xml_parser.h"

#include <functional> 
#include <limits> 
#include <math.h> 

#include <iostream>
#include <fstream>
#include <sstream>

namespace read_msh_ns {

#define ndbbg_read_msh_ns
#define ndebug_set_projector
#define ndebug_match_faces

//--------------------
// check_element_conn
//--------------------
// A map of function pointers used to check element connecivity.
//
// Reproduces 'SUBROUTINE CHECKIEN(lM)' defined in READMSH.f.
//
std::map<consts::ElementType, std::function<void(mshType&)>> check_element_conn = {
  {consts::ElementType::LIN1, check_line_conn},
  {consts::ElementType::HEX8, check_hex8_conn},
  {consts::ElementType::HEX20, check_hex20_conn},
  {consts::ElementType::HEX27, check_hex27_conn},
  {consts::ElementType::QUD4, check_quad4_conn},
  {consts::ElementType::QUD9, check_quad4_conn},
  {consts::ElementType::TET4, check_tet_conn},
  {consts::ElementType::TRI3, check_tri3_conn},
  {consts::ElementType::TRI6, check_tri6_conn},
  {consts::ElementType::TET10, check_tet_conn},
  {consts::ElementType::WDG, check_wedge_conn}
};

//--------------
// calc_elem_ar
//--------------
// Calculate element Aspect Ratio of a given mesh
//
void calc_elem_ar(ComMod& com_mod, const CmMod& cm_mod, mshType& lM, bool& rflag)
{
  using namespace consts;

  if (lM.eType != ElementType::TET4 && lM.eType != ElementType::TRI3) {
    //     wrn = "AR is computed for TRI and TET elements only"
    return; 
  }

  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;

  Vector<double> AsR(lM.nEl);
  Array<double> xl(nsd,lM.eNoN);
  Array<double> dol(nsd,lM.eNoN);
  Vector<int> bins(5);

  for (int e = 0; e < lM.nEl; e++) {
    int iDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto cPhys = com_mod.eq[cEq].dmn[iDmn].phys;

    for (int a = 0; a < lM.eNoN; a++) {
      int Ac = lM.IEN(a,e);
      xl.set_col(a, com_mod.x.col(Ac));
      if (com_mod.mvMsh) {
        for (int i = 0; i < nsd; i++) {
          dol(i,a) = com_mod.Do(i+nsd,Ac);
        }
      }
    }

    if (com_mod.mvMsh) {
      xl = xl + dol;
    }

    AsR(e) = all_fun::aspect_ratio(com_mod, nsd, lM.eNoN, xl);

    double p1 = 0.0;
    double p2 = 5.0;
    int i_index = 0;

    for (int i = 0; i < 5; i++) {
      if (AsR(e) > p1 && AsR(e) <= p2) {
        bins(i) = bins(i) + 1;
        i_index = i;
        break;
      }
      p1 = p2;
      p2 = p1 + 5.0;
    } 

    if (i_index > 3) {
      bins(4) = bins(4) + 1;
    }
  } 

  double maxAR = AsR.max();
  maxAR = com_mod.cm.reduce(cm_mod, maxAR, MPI_MAX);
  bins = com_mod.cm.reduce(cm_mod, bins);

  std::array<double,5> tmp;
  for (int i = 0; i < 5; i++) {
    tmp[i] = 100.0 * static_cast<double>(bins[i]) / static_cast<double>(lM.gnEl);
  }

  if (rflag) {
    /*
    std = "    Max Asp. Ratio <"//maxAR//">"
    std = "    AR [   <  5] <"//bins(1)//">  ("//tmp(1)//".)"
    std = "       [ 5 - 10] <"//bins(2)//">  ("//tmp(2)//".)"
    std = "       [10 - 15] <"//bins(3)//">  ("//tmp(3)//".)"
    std = "       [15 - 20] <"//bins(4)//">  ("//tmp(4)//".)"
    std = "       [   > 20] <"//bins(5)//">  ("//tmp(5)//".)"
    */
  } else {
    //std = "    Max Asp. Ratio <"//maxAR//">"
  }

}

//---------------
// calc_elem_jac
//---------------
// Calculate element Jacobian of a given mesh.
//
void calc_elem_jac(ComMod& com_mod, const CmMod& cm_mod, mshType& lM, bool& rflag)
{
  using namespace consts;
  const int nsd = com_mod.nsd;
  rflag = false; 

  Vector<double> Jac(lM.nEl);
  Array<double> xl(nsd,lM.eNoN);
  Array<double> dol(nsd,lM.eNoN);

  int cEq = com_mod.cEq;
  int cnt = 0;

  for (int e = 0; e < lM.nEl; e++) {
    int iDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto cPhys = com_mod.eq[cEq].dmn[iDmn].phys;

    for (int a = 0; a < lM.eNoN; a++) {
      int Ac = lM.IEN(a,e);
      xl.set_col(a, com_mod.x.col(Ac));
      if (com_mod.mvMsh) {
        for (int i = 0; i < nsd; i++) {
          dol(i,a) = com_mod.Do(i+nsd,Ac);
        }
      }
    }

    if (com_mod.mvMsh) {
      xl = xl + dol;
    }

    Jac(e) = all_fun::jacobian(com_mod, nsd, lM.eNoN, xl, lM.Nx.slice(0));

    if (Jac(e) < 0.0) {
      cnt = cnt + 1;
      if (cPhys != Equation_fluid) {
        throw std::runtime_error("[calc_elem_jac] Negative Jacobian in non-fluid domain.");
      }
    }
  } 

  double maxJ = Jac.abs().max();
  maxJ = com_mod.cm.reduce(cm_mod, maxJ, MPI_MAX);
  Jac = Jac / std::abs(maxJ);

  double minJ = Jac.min();
  cnt = com_mod.cm.reduce(cm_mod, cnt);
  double tmp = 100.0 * static_cast<double>(cnt) / static_cast<double>(lM.gnEl);

  if (minJ < 0.0) {
    rflag = true;
  }

  if (!rflag) {
    //std = "    Min normalized Jacobian <"//minJ//">"
  } else {
    com_mod.rmsh.cntr = com_mod.rmsh.cntr + 1;
    /*
    std = " "
    std = "cccccccccccccccccccccccccccccccccccccccccccccccccc"//
    2  "cccccccc"
    std = ""
    std = " Mesh is DISTORTED (Min. Jacobian < 0) at time "//cTS
    std = "    Min normalized Jacobian <"//minJ//">"
    std = "    No. of Elements with Jac < 0: "//cnt// 2  " ("//tmp//".)"
    */

    if (!com_mod.rmsh.isReqd) {
      throw std::runtime_error("[calc_elem_jac] Unexpected behavior! Mesh is DISTORTED.");
    }
  }
}

//----------------
// calc_elem_skew
//----------------
// Calculate element Skewness of a given mesh.
//
void calc_elem_skew(ComMod& com_mod, const CmMod& cm_mod, mshType& lM, bool& rflag)
{
  using namespace consts;
  const int nsd = com_mod.nsd;

  if (lM.eType != ElementType::TET4 && lM.eType != ElementType::TRI3) {
    std::cout << "WARNING: Skewness is computed for TRI and TET elements only." << std::endl;
    return;
  }

  Vector<double> Skw(lM.nEl);
  Array<double> xl(nsd,lM.eNoN);
  Array<double> dol(nsd,lM.eNoN);
  Vector<int> bins(5);
  int cEq = com_mod.cEq;

  for (int e = 0; e < lM.nEl; e++) {
    int iDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto cPhys = com_mod.eq[cEq].dmn[iDmn].phys;

    for (int a = 0; a < lM.eNoN; a++) {
      int Ac = lM.IEN(a,e);
      xl.set_col(a, com_mod.x.col(Ac));
      if (com_mod.mvMsh) {
        for (int i = 0; i < nsd; i++) {
          dol(i,a) = com_mod.Do(i+nsd,Ac);
        }
      }
    }

    if (com_mod.mvMsh) {
      xl = xl + dol;
    }

    Skw(e) = all_fun::skewness(com_mod, nsd, lM.eNoN, xl);
    double p1 = 0.0;
    double p2 = 0.6;

    for (int i = 0; i < 5; i++) {
      if (Skw(e) > p1 && Skw(e) <= p2) {
        bins[i] = bins[i] + 1;
        break;
      }
      p1 = p2;
      p2 = p1 + 0.1;
    } 
  } 

  double maxSk = Skw.max();
  maxSk = com_mod.cm.reduce(cm_mod, maxSk, MPI_MAX);
  bins = com_mod.cm.reduce(cm_mod, bins);

  std::array<double,5> tmp;
  for (int i = 0; i < 5; i++) {
    tmp[i] = 100.0 * static_cast<double>(bins[i]) / static_cast<double>(lM.gnEl);
  }

  if (rflag) {
    /*
    std = "    Max Skewness <"//maxSk//">"
    std = "    Skew [    < 0.6] <"//bins(1)//">  ("//tmp(1)//".)"
    std = "         [0.6 - 0.7] <"//bins(2)//">  ("//tmp(2)//".)"
    std = "         [0.7 - 0.8] <"//bins(3)//">  ("//tmp(3)//".)"
    std = "         [0.8 - 0.9] <"//bins(4)//">  ("//tmp(4)//".)"
    std = "         [0.9 - 1.0] <"//bins(5)//">  ("//tmp(5)//".)"
    */
  } else {
    //std = "    Max Skewness <"//maxSk//">"
  }

}

//-----------------
// calc_mesh_props
//-----------------
//
void calc_mesh_props(ComMod& com_mod, const CmMod& cm_mod, const int nMesh, std::vector<mshType>& mesh)
{
  using namespace consts;
  auto& rmsh = com_mod.rmsh;

  for (int iM = 0; iM < nMesh; iM++) {
    bool flag;
    calc_elem_jac(com_mod, cm_mod, mesh[iM], flag);

    calc_elem_skew(com_mod, cm_mod, mesh[iM], flag);

    calc_elem_ar(com_mod, cm_mod, mesh[iM], flag);

    rmsh.flag[iM] = flag;
  }

  if (rmsh.isReqd && std::count(rmsh.flag.begin(), rmsh.flag.end(), true) != rmsh.flag.size()) {
    if (com_mod.cTS == rmsh.fTS) {
      rmsh.fTS = rmsh.fTS + rmsh.freq;

      for (int iM = 0; iM < com_mod.nMsh; iM++) {
        for (int e = 0; e < com_mod.msh[iM].nEl; e++) {
          int cDmn = all_fun::domain(com_mod, com_mod.msh[iM], com_mod.cEq, e);
          auto cPhys = com_mod.eq[com_mod.cEq].dmn[cDmn].phys;
          if (cPhys == Equation_fluid) {
            rmsh.flag[iM] = true; 
            break;
          }
        }
      }
      rmsh.cntr = rmsh.cntr + 1;
    }
  }

  Array<bool> gFlag(nMesh, com_mod.cm.np());

  MPI_Allgather(&rmsh.flag, nMesh, cm_mod::mplog, gFlag.data(), nMesh, cm_mod::mplog, com_mod.cm.com());

  for (int iM = 0; iM < nMesh; iM++) {
    rmsh.flag[iM] = false;
    for (int i = 0; i < nMesh; i++) {
      if (gFlag(iM,i)) {
        rmsh.flag[iM] = true;
        break;
      }
    }
  }

  if (std::count(rmsh.flag.begin(), rmsh.flag.end(), true) != 0) {
    com_mod.resetSim = true;
  }
}

//----------
// calc_nbc
//----------
// Checks that face nodes are valid and creates a list of unique 
// node IDs for the face.
//
// Valid face nodes 
//   1) Must be < number of nodes in the volume mesh
//   2) Must match the nodes from its parent volume mesh element 
//
// Face data modified
//   face.gN - Face global node Ids 
//   face.nNo - Number of nodes
//
void calc_nbc(mshType& mesh, faceType& face)
{
  int num_elems = mesh.gnEl;
  int num_nodes = mesh.gnNo;
  auto incNd = Vector<int>(mesh.gnNo);
  face.nNo = 0;

  // Check face nodes.
  //
  std::string msg("[calc_nbc] ");

  for (int e = 0; e < face.nEl; e++ ) {
    int Ec = face.gE(e);
    if (Ec > num_elems) {
      throw std::runtime_error(msg + "element ID " + std::to_string(Ec) + " exceeds the number of elements (" + std::to_string(num_elems) +
          ") in the mesh.");
    } else if (Ec < 0) {
      throw std::runtime_error(msg + "element ID " + std::to_string(Ec) + " is < 0.");
    }

    for (int a = 0; a < face.eNoN; a++) {
      int Ac = face.IEN(a,e);
      if (Ac > num_nodes) { 
        throw std::runtime_error(msg + "element " + std::to_string(e) + " has a node " + std::to_string(Ac) + " that exceeds the number of nodes (" + 
            std::to_string(num_nodes) + ") in the mesh.");
      }
      if (Ac < 0) {
        throw std::runtime_error(msg + "element " + std::to_string(e) + " has a node " + std::to_string(Ac) + " that is < 0.");
      }
      auto elem_conn = mesh.gIEN.col(Ec);
      if (std::find(elem_conn.begin(), elem_conn.end(), Ac) == elem_conn.end()) {
        throw std::runtime_error(msg + "element " + std::to_string(e) + " has a node " + std::to_string(Ac) + 
            " that does not belong to its parent element " + std::to_string(Ec) + ".");
      }

      if (incNd(Ac) == 0) {
        face.nNo += 1;
        incNd(Ac) = 1;
      }
    }
  }

  // Set face nodes.
  face.gN = Vector<int>(face.nNo);
  int a = 0;
  for (int Ac = 0; Ac < mesh.gnNo; Ac++) {
    if (incNd(Ac) != 0) {
      face.gN[a] = Ac;
      a = a + 1;
    } 
  } 
}

//-----------
// check_ien
//-----------
// Check the mesh connectivity and node ordering.
//
// Replicates the Fortran CHECKIEN subroutine defined in READMSH.f.
//
void check_ien(Simulation* simulation, mshType& mesh)
{
  #ifdef debug_check_ien 
  DebugMsg dmsg(__func__, simulation->com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "Mesh name: " << mesh.name;
  dmsg << "Number of nodes (gnNo): " << mesh.gnNo;
  dmsg << "Number of elements (gnEl): " << mesh.gnEl;
  #endif

  // Get the object storing global variables. 
  auto& com_mod = simulation->get_com_mod();
  int nsd = com_mod.nsd;

  Array<double> v(nsd, mesh.eNoN);
  Array<double> xl(nsd, mesh.eNoN);
  Vector<int> incNodes(mesh.gnNo);

  int teNoN = mesh.eNoN;
  int num_nodes = mesh.gnNo;
  int num_elems = mesh.gnEl;
  #ifdef debug_check_ien 
  dmsg << "teNoN: " << teNoN;
  dmsg << "num_elems: " << num_elems;
  #endif

  // Check that the element connectivity has no nodes larger
  // than the number of nodes for the mesh.
  //
  for (int e = 0; e < num_elems; e++) {
    for (int a = 0; a < teNoN; a++) {
      int Ac = mesh.gIEN(a,e);
      if ((Ac >= num_nodes) || (Ac < 0)) {
        throw std::runtime_error("Element " + std::to_string(e) + " contains a node " + std::to_string(Ac) +
            " that is not within 0 and " + std::to_string(num_nodes-1) + "."); 
      }
      incNodes(Ac) = 1;
    }
  }

  // Check that all nodes are referenced by the element connectivity.
  for (int Ac = 0; Ac < num_nodes; Ac++) { 
    if (incNodes(Ac) == 0) {
      throw std::runtime_error("Node " + std::to_string(Ac) + " is not used by any element."); 
    }
  }

  // Check element connectivity node order.
  auto eType = mesh.eType;
  try {
    check_element_conn[eType](mesh);
  } catch (const std::bad_function_call& exception) {
      throw std::runtime_error("[check_ien] eType " + std::to_string(static_cast<int>(eType)) + 
         " is not supported.."); 
  }
}

//----------
// find_blk
//----------
// Compute the block ID for the given coordinate.
//
int find_blk(const int nsd, const int nBkd, const std::vector<bool>& nFlt, const Vector<double>&xMin, const Vector<double>&dx, const Vector<double>& x)
{
  int i = 0;
  int j = 0;
  int k = 0;
  int block_id;

  if (nFlt[0]) {
    i = static_cast<int>((x[0] - xMin[0]) / dx[0]);
  }

  if (i == nBkd) {
    i = nBkd - 1;
  }

  if (nFlt[1]) {
    j = static_cast<int>((x[1] - xMin[1]) / dx[1]);
  }

  if (j == nBkd) {
    j = nBkd - 1;
  }

  if (nsd == 3) {
    if (nFlt[2]) {
      k = static_cast<int>((x[2] - xMin[2]) / dx[2]);
    }
    if (k == nBkd) {
      k = nBkd - 1;
    }
    block_id = k + (j + i*nBkd)*nBkd;
  } else {
    block_id = j + i*nBkd;
  }

  return block_id;
}

//-----------------
// check_line_conn
//-----------------
// Check and reorder line connectivity if needed.
//
// [NOTE] Not implemented.
//
void check_line_conn(mshType& mesh)
{
}

//-----------------
// check_hex8_conn
//-----------------
// Check and reorder hex connectivity if needed.
//
// [NOTE] Not implemented.
//
void check_hex8_conn(mshType& mesh)
{
}

//-----------------
// check_hex20_conn
//-----------------
// Check and reorder hex connectivity if needed.
//
// [NOTE] Not implemented.
//
void check_hex20_conn(mshType& mesh)
{
}

//-----------------
// check_hex27_conn
//-----------------
// Check and reorder hex connectivity if needed.
//
// [NOTE] Not implemented.
//
void check_hex27_conn(mshType& mesh)
{
}

//------------------
// check_quad4_conn 
//------------------
//
// [NOTE] Not implemented.
//
void check_quad4_conn(mshType& mesh)
{
}

//----------------
// check_tet_conn
//----------------
// Check and reorder tet connectivity if needed.
//
void check_tet_conn(mshType& mesh)
{
  Vector<double> v1, v2, v3;
  int num_elems = mesh.gnEl;

  for (int e = 0; e < num_elems; e++) {
    int a = 0, b = 0;
    bool qFlag = false;
    auto elem_conn = mesh.gIEN.col(e);
    auto xl = mesh.x.cols(elem_conn);

    v1 = xl[1] - xl[0];
    v2 = xl[2] - xl[1];
    v3 = xl[3] - xl[2];
    auto v4 = v1.cross(v2);
    int sn = utils::sign(v3.dot(v4));

    if (sn == 1) {
      a = 0;
      b = 1;
      qFlag = true;
      if (e == 0) {
        //std::cout << "[check_tet_conn] Reorder element connectivity." << std::endl;
      }
    } else if (sn == 0) { 
      throw std::runtime_error("Tet element " + std::to_string(e) + " is distorted.");
    } 

    utils::swap( mesh.gIEN(a,e), mesh.gIEN(b,e));

    if (qFlag) {
      if (mesh.eType == consts::ElementType::TET10) {
        a = 5; b = 6;
        utils::swap(mesh.gIEN(a,e), mesh.gIEN(b,e));
        a = 7; b = 8;
        utils::swap(mesh.gIEN(a,e), mesh.gIEN(b,e));
      }
    }
  }
}

//-----------------
// check_tri3_conn
//-----------------
// [NOTE] Not implemented.
//
void check_tri3_conn(mshType& mesh)
{
}

//-----------------
// check_tri6_conn
//-----------------
// [NOTE] Not implemented.
//
void check_tri6_conn(mshType& mesh)
{
}

//------------------
// check_wedge_conn
//------------------
// Check and reorder wedge connectivity if needed.
//
void check_wedge_conn(mshType& mesh)
{
  Vector<double> v1, v2, v3;
  int num_elems = mesh.gnEl;

  for (int e = 0; e < num_elems; e++) {
    int a = 0, b = 0;
    bool qFlag = false;
    auto elem_conn = mesh.gIEN.col(e);
    auto xl = mesh.x.cols(elem_conn);
    v1 = xl[1] - xl[0];
    v2 = xl[2] - xl[1];
    v3 = xl[3] - xl[0];
    auto v4 = v1.cross(v2);
    int sn = utils::sign(v3.dot(v4));

    if (sn == -1) {
      utils::swap( mesh.gIEN(0,e), mesh.gIEN(1,e));
      a = 4;
      b = 5;
      if (e == 0) {
        //std::cout << "[check_wedge_conn] Reorder element connectivity." << std::endl;
      }
    } else if (sn == 0) { 
      throw std::runtime_error("Element " + std::to_string(e) + " is distorted.");
    } 

    utils::swap( mesh.gIEN(a,e), mesh.gIEN(b,e));
  }
}

//--------------
// load_var_ini
//--------------
// Read initial field values (pressure, velocity or displacement).
//
// Variables that may be changed
//   com_mod.Pinit - 
//   com_mod.Vinit - 
//   com_mod.Dinit - 
//
// Reproduces Fortran 'LOADVARINI'.
//
void load_var_ini(Simulation* simulation, ComMod& com_mod)
{
  // Initialize mesh pressure from a file.
  //
  bool flag = false;
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto mesh_param = simulation->parameters.mesh_parameters[iM];

    if (mesh_param->initial_pressures_file_path.defined()) {
      flag = true;
      com_mod.msh[iM].x.resize(1, com_mod.msh[iM].gnNo);
      auto cTmp = mesh_param->initial_pressures_file_path.value();
      int data_comp = 1; 
      int data_series = 0; 
      vtk_xml::read_vtu_pdata(cTmp, "Pressure", com_mod.nsd, data_comp, data_series, com_mod.msh[iM]);
    }
  }

  if (flag) { 
    com_mod.Pinit.resize(com_mod.gtnNo);
    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      if (com_mod.msh[iM].x.size() == 0) { 
        continue;
      }
      for (int a = 0; a < com_mod.msh[iM].gnNo; a++) {
        int Ac = com_mod.msh[iM].gN[a];
        com_mod.Pinit[Ac] = com_mod.msh[iM].x(0,a);
      }
      com_mod.msh[iM].x.clear();
    }
  }

  // Initialize mesh velocities a from file.
  //
  flag = false;

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto mesh_param = simulation->parameters.mesh_parameters[iM];
    if (mesh_param->initial_velocities_file_path.defined()) {
      flag = true;
      com_mod.msh[iM].x.resize(com_mod.nsd, com_mod.msh[iM].gnNo);
      auto cTmp = mesh_param->initial_velocities_file_path.value();
      int data_comp = com_mod.nsd; 
      int data_series = 0; 
      vtk_xml::read_vtu_pdata(cTmp, "Velocity", com_mod.nsd, data_comp, data_series, com_mod.msh[iM]);
    }
  }

  if (flag) {
    com_mod.Vinit.resize(com_mod.nsd,com_mod.gtnNo);
    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      if (com_mod.msh[iM].x.size() == 0) {
        continue;
      }
      for (int a = 0; a < com_mod.msh[iM].gnNo; a++) {
        int Ac = com_mod.msh[iM].gN[a];
        for (int i = 0; i < com_mod.nsd; i++) {
          com_mod.Vinit(i,Ac) = com_mod.msh[iM].x(i,a);
        }
      }
      com_mod.msh[iM].x.clear();
    }
  }

  // Initialize mesh displacements a from file.
  //
  flag = false;

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto mesh_param = simulation->parameters.mesh_parameters[iM];
    if (mesh_param->initial_displacements_file_path.defined()) {
      flag = true;
      com_mod.msh[iM].x.resize(com_mod.nsd, com_mod.msh[iM].gnNo);
      auto cTmp = mesh_param->initial_displacements_file_path.value();
      int data_comp = com_mod.nsd; 
      int data_series = 0; 
      vtk_xml::read_vtu_pdata(cTmp, "Displacement", com_mod.nsd, data_comp, data_series, com_mod.msh[iM]);
    }
  }

  if (flag) {
    com_mod.Dinit.resize(com_mod.nsd,com_mod.gtnNo);
    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      if (com_mod.msh[iM].x.size() == 0) {
        continue;
      }
      for (int a = 0; a < com_mod.msh[iM].gnNo; a++) {
        int Ac = com_mod.msh[iM].gN[a];
        for (int i = 0; i < com_mod.nsd; i++) {
          com_mod.Dinit(i,Ac) = com_mod.msh[iM].x(i,a);
        }
      }
      com_mod.msh[iM].x.clear();
    }
  }
}

//-------------
// match_faces
//-------------
// Match isoparameteric faces to each other. 
//
// Project nodes from two adjacent meshes to each other based on a L2 norm.
//
void match_faces(const ComMod& com_mod, const faceType& lFa, const faceType& pFa, const double ptol, utils::stackType& lPrj)
{
  int iM  = lFa.iM;
  int iSh = 0;
  for (int i = 0; i < iM; i++) {
    iSh = iSh + com_mod.msh[i].gnNo;
  }

  int jM  = pFa.iM;
  int jSh = 0;
  for (int j = 0; j < jM; j++) {
    jSh = jSh + com_mod.msh[j].gnNo;
  }

  double tol;
  double eps = std::numeric_limits<double>::epsilon();

  if (utils::is_zero(ptol)) {
    tol = 1.e3 * eps;
  } else { 
    tol = ptol;
  }

  // We want to have approximately 1000 nodes in each block. So we
  // calculate nBkd, which is the number of separate blockes in each
  // direction, based on that.
  //
  int a = pFa.nNo;
  int nBkd = static_cast<int>( pow(a/1000.0, 0.333) + 0.5) ;
  if (nBkd == 0) {
    nBkd = 1;
  }
  int nsd = com_mod.nsd;;
  int nBk = pow(nBkd, nsd);

  // Find the extents of the domain and size of each block.
  //
  auto lfa_nodes = lFa.gN + iSh;
  auto pfa_nodes = pFa.gN + jSh;
  Vector<double> xMin(com_mod.nsd), xMax(com_mod.nsd);

  for (int i = 0; i < nsd; i++) {
    auto lfa_coords = com_mod.x.rows(i, lfa_nodes);
    auto pfa_coords = com_mod.x.rows(i, pfa_nodes);
    xMin[i] = std::min(lfa_coords.min(), pfa_coords.min());
    xMax[i] = std::max(lfa_coords.max(), pfa_coords.max());

    if (xMin[i] < 0.0) {
      xMin[i] = xMin[i]*(1.0+eps);
    } else { 
      xMin[i] = xMin[i]*(1.0-eps);
    } 

    if (xMax[i] < 0.0) { 
      xMax[i] = xMax[i]*(1.0-eps);
    } else { 
      xMax[i] = xMax[i]*(1.0+eps);
    } 
  }

  auto dx = (xMax - xMin) / static_cast<double>(nBkd);
  std::vector<bool> nFlt(nsd);

  for (int i = 0; i < nsd; i++) {
    if (utils::is_zero(dx[i])) {
      nFlt[i] = false;
    } else {
      nFlt[i] = true;
    } 
  }
  
  Vector<int> nodeBlk(a); 
  std::vector<blkType> blk(nBk);

  // Find an estimation for size of each block
  //
  for (int a = 0; a < pFa.nNo; a++) {
    int Ac  = pFa.gN[a] + jSh;
    auto coord = com_mod.x.col(Ac);
    int iBk = find_blk(nsd, nBkd, nFlt, xMin, dx, coord);
    nodeBlk[a] = iBk;
    blk[iBk].n = blk[iBk].n + 1;
  }

  for (int iBk = 0; iBk < nBk; iBk++) {
    blk[iBk].gN = Vector<int>(blk[iBk].n);
    blk[iBk].n = 0;
  }

  for (int a = 0; a < pFa.nNo; a++) {
    int Ac = pFa.gN[a];
    int iBk = nodeBlk[a];
    blk[iBk].gN(blk[iBk].n) = Ac;
    blk[iBk].n = blk[iBk].n + 1;
  } 

  // Doing the calculation for every single node on this face.
  //
  int cnt  = 0;

  for (int a = 0; a < lFa.nNo; a++) {
    int Ac  = lFa.gN[a];
    auto coord = com_mod.x.col(Ac+iSh);
    int iBk = find_blk(nsd, nBkd, nFlt, xMin, dx, coord);

    // Check all nodes on the other face.
    auto minS = std::numeric_limits<double>::max();
    int i;

    for (int b = 0; b < blk[iBk].n; b++) {
      int Bc = blk[iBk].gN[b];
      if ((iM == jM) && (Ac == Bc)) {
        continue;
      }

      auto diff = com_mod.x.col(Bc+jSh) - com_mod.x.col(Ac+iSh);
      double ds = sqrt(diff*diff); 

      if (ds < minS) { 
        minS = ds;
        i = Bc;
      }
    }

    int Bc = i;

    if (tol < 0.0) {
      push_stack(lPrj, {Ac, Bc});
      cnt = cnt + 1;

    } else if (minS < tol) {
      push_stack(lPrj, {Ac, Bc});
      cnt = cnt + 1;
    }
  }

  if (cnt != lFa.nNo) {
    throw std::runtime_error("Failed to find matching nodes between faces " + lFa.name + " and " + pFa.name + ".");
  }

  if (lPrj.n == 0) {
    throw std::runtime_error("No matching nodes between faces " + lFa.name + " and " + pFa.name + ".");
  }

  if (lPrj.n/2 != lFa.nNo) {
    throw std::runtime_error("Mismatch of nodes between faces " + lFa.name + " and " + pFa.name + ".");
  }

} 

//--------------
// read_fib_nff
//--------------
// Read fiber direction from a vtu file.
//
// The fiber orientations data is copied into mesh.fN. 
//
void read_fib_nff(Simulation* simulation, mshType& mesh, const std::string& fName, const std::string& kwrd, const int idx)
{
  vtk_xml_parser::load_fiber_direction_vtu(fName, kwrd, idx, simulation->com_mod.nsd, mesh);
}

//----------
// read_msh
//----------
// For each mesh defined for the simulation 
//
//   1) Set mesh parameters 
//
//   2) Read mesh nodal coordiantes and element connectivity 
//
void read_msh(Simulation* simulation)
{
  auto& com_mod = simulation->get_com_mod();

  #define n_debug_read_msh 
  #ifdef debug_read_msh 
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg <<  "resetSim: " << com_mod.resetSim;
  #endif

  // Allocate global 'msh' vector of mshType objects.
  //
  // READMSH - ALLOCATE (msh(nMsh), gX(0,0))
  //
  com_mod.nMsh = simulation->parameters.mesh_parameters.size();
  com_mod.msh.resize(com_mod.nMsh);
  #ifdef debug_read_msh 
  dmsg << "Number of meshes: " << com_mod.nMsh;
  #endif

  double max_dval = std::numeric_limits<double>::max();
  double min_dval = std::numeric_limits<double>::min();
  std::array<double,3> min_x{min_dval, min_dval, min_dval};
  std::array<double,3> max_x{max_dval, max_dval, max_dval};

  // [NOTE] We are not processing 'resetSim' here, not sure what that does..
  //
  // I guess we should put all of the follow code in a function.
  /*
  IF (.NOT.resetSim) THEN
  */

  // Global total number of nodes.
  com_mod.gtnNo = 0;

  // Set mesh parameters and read mesh data.
  //
  // READMSH - lPtr => lPM%get(msh(iM)%lShl,"Set mesh as shell") 
  //
  // READMSH - CALL READSV(lPM, msh(iM)) 
  //
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto param = simulation->parameters.mesh_parameters[iM];
    auto& mesh = com_mod.msh[iM];

    mesh.name = param->name();
    mesh.lShl = param->set_mesh_as_shell();
    mesh.lFib = param->set_mesh_as_fibers();
    mesh.scF = param->mesh_scale_factor();
    #ifdef debug_read_msh 
    dmsg << "Mesh name: " << mesh.name;
    dmsg << "  mesh.lShl: " << mesh.lShl;
    dmsg << "  mesh.lFib: " << mesh.lFib;
    dmsg << "  scale factor: " << mesh.scF;
    #endif

    // Read mesh nodal coordinates and element connectivity.
    load_msh::read_sv(simulation, mesh, param);

    // [NOTE] What is this all about?
    if (mesh.eType == consts::ElementType::NA) {
      load_msh::read_ccne(simulation, mesh, param);
    }

    // Allocate mesh local to global nodes map (gN).
    #ifdef debug_read_msh 
    dmsg << "mesh.gnNo: " << mesh.gnNo;
    #endif
    mesh.gN = Vector<int>(mesh.gnNo);
    mesh.gN = -1;

    // Check for unique face names.
    //
    #ifdef debug_read_msh 
    dmsg << "Check for unique face names ....";
    #endif
    for (int iFa = 0; iFa < mesh.nFa; iFa++) {
      mesh.fa[iFa].iM = iM;
      auto ctmp = mesh.fa[iFa].name;
      for (int i = 0; i < iM; i++) {
        for (int j = 0; j < com_mod.msh[i].nFa; j++) {
          if ((ctmp == com_mod.msh[i].fa[j].name) && ((i != iM || j != iFa))) { 
            throw std::runtime_error("The face name '" + ctmp + "' is duplicated."); 
          }
        }
      } 
    } 

    // Global total number of nodes.
    com_mod.gtnNo += mesh.gnNo;
    #ifdef debug_read_msh 
    dmsg << "Global total number of nodes (gtnNo): " << com_mod.gtnNo;
    #endif
  }

  // Create global nodal coordinate array. 
  //
  // Scale the nodal coordinates. 
  //
  auto gX = Array<double>(com_mod.nsd, com_mod.gtnNo);
  int n = 0;

  for (auto& mesh : com_mod.msh) {
    for (int i = 0; i < mesh.gnNo; i++) {
      for (int j = 0; j < com_mod.nsd; j++) {
        gX(j,n) = mesh.scF * mesh.x(j,i); 
      }
    n += 1;
    }
    mesh.x.clear();
  }

  com_mod.x = gX;

  // Check for shell elements.
  //
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto& mesh = com_mod.msh[iM];
    if (mesh.lShl) {
      if (mesh.eType != consts::ElementType::NRB && mesh.eType != consts::ElementType::TRI3) {
        throw std::runtime_error("Shell elements must be triangles or C1-NURBS.");
      }
      if (mesh.eType == consts::ElementType::NRB) {
        for (int i = 0; i < com_mod.nsd-1; i++) {
          if (mesh.bs[i].p <= 1) {
            throw std::runtime_error("NURBS for shell elements must have be p > 1.");
          } 
        } 
      } 
    } 
  } 

  // Check for fiber mesh.
  //
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto& mesh = com_mod.msh[iM];
    if (mesh.lFib) {
      if (mesh.eType != consts::ElementType::LIN1 && mesh.eType != consts::ElementType::LIN2) { 
        throw std::runtime_error("Fiber elements must be either linear quadratic.");
      }
    }
  }
  
  // [NOTE] We are not processing resetSim here.
  /*
  ELSE
    ALLOCATE(gX(nsd,gtnNo))
    gX = x
    DO iM=1, nMsh
      CALL SELECTELE(msh(iM))
      ALLOCATE(msh(iM)%gN(msh(iM)%gnNo))
      msh(iM)%gN = 0
    END DO
  END IF ! resetSim
*/

  // Examining the existance of projection faces and setting %gN.
  // Reseting gtnNo and recounting nodes that are not duplicated
  //
  com_mod.gtnNo = 0;
  utils::stackType avNds;

  set_projector(simulation, avNds);

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto& mesh = com_mod.msh[iM];
    for (int a = 0; a < mesh.gnNo; a++) {
      if (mesh.gN[a] == -1) { 
        int i;
        if (pull_stack(avNds,i)) {
          mesh.gN[a] = i;
        } else { 
          mesh.gN[a] = com_mod.gtnNo;
          com_mod.gtnNo = com_mod.gtnNo + 1;
        }
      }
      if (mesh.gpN.size() != 0) {
        mesh.gpN[a] = mesh.gN[a];
      }
    }
  }

  com_mod.x = Array<double>(com_mod.nsd, com_mod.gtnNo);

  if (avNds.n != 0) {
    throw std::runtime_error("There are " + std::to_string(avNds.n) + " nodes not associated other faces.");
  }

  #ifdef debug_read_msh 
  dmsg << "com_mod.gtnNo: " << com_mod.gtnNo;
  #endif

  // Temporarily allocate msh%lN array. This is necessary for BCs and
  // will later be deallocated in DISTRIBUTE
  //
  #ifdef debug_read_msh 
  dmsg << "Temporarily allocate msh%lN array ... " ;
  #endif

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    com_mod.msh[iM].lN = Vector<int>(com_mod.gtnNo);
    com_mod.msh[iM].lN = -1;
    for (int a = 0; a < com_mod.msh[iM].gnNo; a++ ) {
      int Ac = com_mod.msh[iM].gN[a];
      com_mod.msh[iM].lN[Ac] = a;
    }
  }

  // Re-arranging x and finding the size of the entire domain
  // First rearrange 2D/3D mesh and then, 1D fiber mesh
  //
  #ifdef debug_read_msh 
  dmsg << "Re-arranging x ..." ;
  #endif

  std::vector<bool> ichk(com_mod.gtnNo);
  std::fill(ichk.begin(), ichk.end(), false);
  int b = 0;

  Vector<double> maxX(3), minX(3);
  minX = std::numeric_limits<double>::max();
  maxX = -std::numeric_limits<double>::max();

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    if (com_mod.msh[iM].lFib) {
      b = b + com_mod.msh[iM].gnNo;
      continue;
    }

    for (int a = 0; a < com_mod.msh[iM].gnNo; a++) {
      int Ac = com_mod.msh[iM].gN[a];
      ichk[Ac] = true;

      for (int i = 0; i < com_mod.nsd; i++) {
        com_mod.x(i,Ac) = gX(i,b);
        if (maxX[i] < com_mod.x(i,Ac)) {
          maxX[i] = com_mod.x(i,Ac);
        }
        if (minX[i] > com_mod.x(i,Ac)) {
          minX[i] = com_mod.x(i,Ac);
        }
      } 

      b  = b + 1;
    } 
  } 

  b = 0;
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    if (!com_mod.msh[iM].lFib) {
      b = b + com_mod.msh[iM].gnNo;
      continue;
    }

    for (int a = 0; a < com_mod.msh[iM].gnNo; a++) {
      int Ac = com_mod.msh[iM].gN[a];
      if (ichk[Ac]) {
        continue; 
      }
      for (int i = 0; i < com_mod.nsd; i++) {
        com_mod.x(i,Ac) = gX(i,b);
        if (maxX[i] < com_mod.x(i,Ac)) {
          maxX[i] = com_mod.x(i,Ac);
        }
        if (minX[i] > com_mod.x(i,Ac)) {
          minX[i] = com_mod.x(i,Ac);
        }
      } 

      b  = b + 1;
    }
  }

  #ifdef debug_read_msh 
  dmsg << "minX: " << minX;
  dmsg << "maxX: " << maxX;
  #endif

  // Rearrange fa
  //
  #ifdef debug_read_msh 
  dmsg << "Rearrange " << " fa ... ";
  #endif

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    for (int iFa = 0; iFa < com_mod.msh[iM].nFa; iFa++) {
      for (int a = 0; a < com_mod.msh[iM].fa[iFa].nNo; a++) {
        int Ac = com_mod.msh[iM].fa[iFa].gN[a];
        Ac = com_mod.msh[iM].gN[Ac];
        com_mod.msh[iM].fa[iFa].gN[a] = Ac;
      }     

      for (int e = 0; e < com_mod.msh[iM].fa[iFa].nEl; e++) {
        for (int a = 0; a < com_mod.msh[iM].fa[iFa].eNoN; a++) {
          int Ac = com_mod.msh[iM].fa[iFa].IEN(a,e);
          Ac = com_mod.msh[iM].gN[Ac];
          com_mod.msh[iM].fa[iFa].IEN(a,e) = Ac;
        }      
      } 
    }  
  }  

  // [NOTE] Not implemented. 
/*
  if (resetSim) {
    int lDof = size(rmsh%Y0,1)
    int lnNo = size(rmsh%Y0,2)
    if (lnNo != gtnNo) {
      ALLOCATE(tmpA(lDof,lnNo))
      ALLOCATE(tmpY(lDof,lnNo))
      ALLOCATE(tmpD(lDof,lnNo))
      tmpA = rmsh%A0
      tmpY = rmsh%Y0
      tmpD = rmsh%D0
      DEALLOCATE(rmsh%A0,rmsh%Y0,rmsh%D0)
      ALLOCATE(rmsh%A0(lDof,gtnNo))
      ALLOCATE(rmsh%Y0(lDof,gtnNo))
      ALLOCATE(rmsh%D0(lDof,gtnNo))
      int b = 0
      DO iM=1, nMsh
        DO a=1, msh(iM)%gnNo
          b = b + 1
          Ac = msh(iM)%gpN(a)
          DO i=1, lDof
            rmsh%A0(i,Ac) = tmpA(i,b)
            rmsh%Y0(i,Ac) = tmpY(i,b)
            rmsh%D0(i,Ac) = tmpD(i,b)
          END DO
        END DO
      END DO
    END IF ! lnNo < gtnNo
  END IF ! resetSim
*/

  // Setting dmnId parameter here, if there is at least one mesh that
  // has defined eId.
  //
  #ifdef debug_read_msh 
  dmsg << "Setting dmnId parameter ... " << std::endl;
  #endif
  bool flag = false;

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    #ifdef debug_read_msh 
    dmsg << " ----- iM " << iM << " -----";
    #endif
    auto mesh_param = simulation->parameters.mesh_parameters[iM];

    if (mesh_param->domain_id.defined()) { 
      int domain_id = mesh_param->domain_id();
      #ifdef debug_read_msh 
      dmsg << "domain_id: " << domain_id;
      #endif
      all_fun::set_dmn_id(com_mod.msh[iM], domain_id);
    }

    // Read in domain IDs. 
    //
    if (mesh_param->domain_file_path.defined()) { 
      /*
      if (rmsh.isReqd) {
        throw std::runtime_error("Variable domain properties are not allowed with remeshing.");
      }
      */
      auto domain_file_path = mesh_param->domain_file_path.value(); 
      #ifdef debug_read_msh 
      dmsg << "domain_file_path: " << domain_file_path;
      #endif
      if ((domain_file_path.find(".vtp") != std::string::npos) || (domain_file_path.find(".vtu") != std::string::npos)) {
        set_dmn_id_vtk(simulation, com_mod.msh[iM], domain_file_path, "DOMAIN_ID");
      } else {
        set_dmn_id_ff(simulation, com_mod.msh[iM], domain_file_path);
      }
    }

    if (com_mod.msh[iM].eId.size() != 0) {
      flag = true;
    }
  }

  if (flag) {
    auto& dmnId = com_mod.dmnId;
    dmnId.resize(com_mod.gtnNo);

    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      if (com_mod.msh[iM].eId.size() == 0) {
        continue; 
      }
      for (int e = 0; e < com_mod.msh[iM].gnEl; e++) {
        for (int a = 0; a < com_mod.msh[iM].eNoN; a++) {
          int Ac = com_mod.msh[iM].gIEN(a,e);
          Ac = com_mod.msh[iM].gN[Ac];
          dmnId[Ac] = dmnId[Ac] | com_mod.msh[iM].eId[e];
        }
      }
    }
  }

  // Read fiber orientation.
  //
  flag = false;

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto mesh_param = simulation->parameters.mesh_parameters[iM];
    int num_paths = mesh_param->fiber_direction_file_paths.size();
    int num_dirs = mesh_param->fiber_directions.size();
    if ((num_paths != 0) || (num_dirs != 0)) {
      flag = true;
      break; 
    }
  }

  int nsd = com_mod.nsd;

  if (flag) {
    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      auto mesh_param = simulation->parameters.mesh_parameters[iM];
      int num_paths = mesh_param->fiber_direction_file_paths.size();
      if (num_paths != 0) {
        com_mod.msh[iM].nFn = num_paths;
        int num_elems = com_mod.msh[iM].gnEl;
        com_mod.msh[iM].fN = Array<double>(num_paths*nsd, num_elems);
        com_mod.msh[iM].fN = 0.0;
        auto fiber_paths = mesh_param->fiber_direction_file_paths();

        // Read fiber directions from vtk format files.
        for (int i = 0; i < com_mod.msh[iM].nFn; i++) {
          auto cTmp = fiber_paths[i];
          read_fib_nff(simulation, com_mod.msh[iM], cTmp, "FIB_DIR", i);
        }
      } else { 
        int num_dirs = mesh_param->fiber_directions.size();

        if (num_dirs != 0) {
          com_mod.msh[iM].nFn = num_dirs; 
          int num_elems = com_mod.msh[iM].gnEl;
          com_mod.msh[iM].fN = Array<double>(num_dirs*nsd, num_elems);
          com_mod.msh[iM].fN = 0.0;
          auto fiber_dirs = mesh_param->fiber_directions;

          for (int i = 0; i < com_mod.msh[iM].nFn; i++) {
            auto fibN = fiber_dirs[i]();
            double rtmp = sqrt(fibN[0]*fibN[0] + fibN[1]*fibN[1] + fibN[2]*fibN[2]);
            if (!utils::is_zero(rtmp)) {
              fibN[0] = fibN[0] / rtmp;
              fibN[1] = fibN[1] / rtmp;
              fibN[2] = fibN[2] / rtmp;
            }
            for (int e = 0; e < com_mod.msh[iM].gnEl; e++) {
              for (int j = 0; j < nsd; j++) {
                com_mod.msh[iM].fN(i*nsd+j,e) = fibN[j];
              }
            }
          }
        }
      }
    }
  }

  // Read prestress data.
  //
  flag = false;

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto mesh_param = simulation->parameters.mesh_parameters[iM];
    if (mesh_param->prestress_file_path.defined()) {
      /* [NOTE] Not implemented.
      if (rmsh.isReqd) {
        throw std::runtime_error("Prestress is currently not allowed with remeshing.");
      }
      */
      auto cTmp = mesh_param->prestress_file_path.value(); 
      flag = true;
      com_mod.msh[iM].x = Array<double>(com_mod.nsymd, com_mod.msh[iM].gnNo);
      vtk_xml::read_vtu_pdata(cTmp, "Stress", com_mod.nsd, com_mod.nsymd, 0, com_mod.msh[iM]);
    }
  }

  if (flag) {
    com_mod.pS0 = Array<double>(com_mod.nsymd, com_mod.gtnNo);

    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      if (com_mod.msh[iM].x.size() == 0) {
        continue; 
      } 
      for (int a = 0; a < com_mod.msh[iM].gnNo; a++) {
        int Ac = com_mod.msh[iM].gN[a];
        for (int i = 0; i < com_mod.nsymd; i++) {
          com_mod.pS0(i,Ac) = com_mod.msh[iM].x(i,a);
        }
      }
      com_mod.msh[iM].x.clear();
    }
  }

  // Set initial mesh pressure, velocity or displacement from a file.
  if (!com_mod.resetSim) {
    load_var_ini(simulation, com_mod);
  }

  // Print mesh statistics. 
  //
  if (com_mod.nMsh > 1) {
    int num_elems = 0;
    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      num_elems += com_mod.msh[iM].gnEl;
    }
  }

  // Read contact model parameters.
  //
  // [NOTE] Implement this, no tests yet.
  //
  if (!com_mod.resetSim) {
    com_mod.iCntct = false;
    //lPM => list%get(ctmp, "Contact model")
    /*
    IF (ASSOCIATED(lPM)) THEN
      iCntct = .TRUE.
      SELECT CASE (TRIM(ctmp))
      CASE ("penalty")
        cntctM%cType = cntctM_penalty
        lPtr => lPM%get(cntctM%k,
     2            "Penalty constant (k)", 1, ll=0._RKIND)
        lPtr => lPM%get(cntctM%h,
     2            "Desired separation (h)", 1, lb=0._RKIND)
        lPtr => lPM%get(cntctM%c,
     2            "Closest gap to activate penalty (c)", 1, lb=0._RKIND)
        IF (cntctM%c .LT. cntctM%h) err =
     2            "Choose c > h for proper contact penalization"
        lPtr => lPM%get(cntctM%al,
     2            "Min norm of face normals (alpha)",1,lb=0._RKIND,
     3  ub=1._RKIND)
      CASE DEFAULT
        err = "Undefined contact model"
      END SELECT
    END IF
    */
  }
}

//---------------
// set_dmn_id_ff 
//---------------
// Read domain from a dat file
//
// Reproduces the Fortran 'SETDMNIDFF' subroutine.
//
void set_dmn_id_ff(Simulation* simulation, mshType& lM, const std::string& file_name)
{
  #define n_debug_set_dmn_id_ff
  #ifdef debug_set_dmn_id_ff 
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "file_name: " << file_name;
  dmsg << "lM.gnEl: " << lM.gnEl;
  #endif

  int btSiz = std::numeric_limits<int>::digits + 1;

  // Check to see if I need to increase the size of dmnId
  //
  if (lM.eId.size() == 0) { 
    lM.eId.resize(lM.gnEl);
  }

  Vector<int> iDmn(btSiz);

  std::ifstream domain_ids_file;
  domain_ids_file.open(file_name);
  if (!domain_ids_file.is_open()) {
    throw std::runtime_error("Failed to open the domain IDs file '" + file_name + "'.");
  }

  // Read the integer domain IDs.
  //
  int domain_id;
  std::string str_value;

  for (int e = 0; e < lM.gnEl; e++) {
    if (!domain_ids_file.good()) {
      throw std::runtime_error("The domain IDs file '" + file_name + "' contains too few values than the number of elements.");
      break;
    }

    domain_ids_file >> str_value;
    std::istringstream str_stream(str_value);

    if (!(str_stream >> domain_id)) {
      throw std::runtime_error("Domain ID integer value '" + str_value + "' found in the domain IDs file '" + file_name + "'.");
    } 

    if (domain_id <= 0) {
      throw std::runtime_error("The domain ID '" + str_value + "' found in the domain IDs file '" + file_name + "' is <= 0.");
    }

    int num_bits = utils::CountBits(domain_id);

    if (num_bits > btSiz) {
      throw std::runtime_error("The domain ID '" + str_value + "' found in domain IDs file '" + file_name + "' is too large.");
    }

    lM.eId(e) = lM.eId(e) | 1UL << domain_id;
  }
}

//----------------
// set_dmn_id_vtk
//----------------
// Read mesh domains from a vtu/vtp file.
//
// [NOTE] Not implemented.
//
void set_dmn_id_vtk(Simulation* simulation, mshType& mesh, const std::string& file_name, const std::string& kwrd)
{
  int btSiz = std::numeric_limits<int>::digits;
}

//---------------
// set_projector
//---------------
// This routines associates two faces with each other and sets gN.
//
// Data set
//   mesh.gpN = Vector<int>(gnNo);
//   com_mod.msh[].gpN = Vector<int>(gnNo);
//   com_mod.gtnNo 
//   com_mod.msh[].gN[]
//
void set_projector(Simulation* simulation, utils::stackType& avNds)
{
  auto& com_mod = simulation->get_com_mod();
  int nPrj = simulation->parameters.projection_parameters.size();

  if (nPrj > 0) {
    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      auto& mesh = com_mod.msh[iM];
      int gnNo = mesh.gnNo;
      mesh.gpN = Vector<int>(gnNo);
    }
  }

  // Calculate an upper limit for the number required stacks
  //
  int nStk = 0;
  for (auto& params : simulation->parameters.projection_parameters) { 
    auto ctmpi = params->name();
    int iM, iFa;
    all_fun::find_face(com_mod.msh, ctmpi, iM, iFa);
    nStk = nStk + com_mod.msh[iM].fa[iFa].nNo;
  }
  std::vector<utils::stackType> stk(nStk);
  utils::stackType lPrj;

  // Match the nodal coordinates for each projection face.
  //
  for (auto& params : simulation->parameters.projection_parameters) { 
    int iM, iFa;
    auto ctmpi = params->name();
    all_fun::find_face(com_mod.msh, ctmpi, iM, iFa);
    auto& face1 = com_mod.msh[iM].fa[iFa];

    int jM, jFa;
    auto ctmpj = params->project_from_face();
    all_fun::find_face(com_mod.msh, ctmpj, jM, jFa);
    auto& face2 = com_mod.msh[jM].fa[jFa];

    double tol = params->projection_tolerance();

    // Match face nodes?
    match_faces(com_mod, face1, face2, tol, lPrj);

    while (true) {
      int ia, ja, i, j, k;

      if (!utils::pull_stack(lPrj,ja)) {
        break;
      }

      if (!utils::pull_stack(lPrj,ia)) {
        break;
      }

      i = com_mod.msh[iM].gN[ia];
      j = com_mod.msh[jM].gN[ja];

      if (i == -1) {
        if (j == -1) {
          // Since neither of them have value add a new node and both of them to the stack.
          if (!utils::pull_stack(avNds, k)) {
            k = com_mod.gtnNo;
            com_mod.gtnNo = com_mod.gtnNo + 1;
          }
          com_mod.msh[iM].gN[ia] = k;
          com_mod.msh[jM].gN[ja] = k;
          push_stack(stk[k], {iM,ia,jM,ja});

        // This is the case one of them has already been assigned. So just using that value for the other one
        } else { 
          com_mod.msh[iM].gN[ia] = j;
          push_stack(stk[j], {iM,ia});
        }
      } else { 
        if (j == -1) {
          com_mod.msh[jM].gN[ja] = i;
          push_stack(stk[i], {jM,ja});

        // Since they are both already have assigned values, I will move the
        // nodes from stack with bigger ID, j, to the other stack, i.
        } else { 
          if (i == j) {
            continue; 
          }
          if (i > j) {
            k = i;
            i = j;
            j = k;
          } 
          while (true) {
            int kM;
            if (!utils::pull_stack(stk[j],ja)) {
              break; 
            }
            if (!utils::pull_stack(stk[j],kM)) { 
              break; 
            }
            com_mod.msh[kM].gN[ja] = i;
            push_stack(stk[i], {kM,ja});
          } 
          utils::push_stack(avNds, j);
        } 
      } 
    }
  }
}

};
