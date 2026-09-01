#include "atom_data_packet.h"
#include "common/rbmd_define.h"

/**
 * @brief 为 AtomicAtomDataPacket创建并提交 MPI 派生数据类型。
 */
void CreateMpiAtomicAtomDataPacket(MPI_Datatype* new_type) {
    // 成员数量更新为 8
    const int count = 8;

    // 块长度（每个成员都是单个元素）
    int blocklengths[count] = {1, 1, 1, 1, 1, 1, 1, 1};

    // 成员的 MPI 数据类型
    MPI_Datatype types[count] = {
        MPI_RBMD_ID,   // atom_id
        MPI_RBMD_ID,   // atom_type
        MPI_RBMD_REAL, // px
        MPI_RBMD_REAL, // py
        MPI_RBMD_REAL, // pz
        MPI_RBMD_REAL, // vx
        MPI_RBMD_REAL, // vy
        MPI_RBMD_REAL  // vz
    };

    // 内存位移
    MPI_Aint displacements[count];

    // 获取地址和计算位移
    AtomicAtomDataPacket packet_instance = {};
    MPI_Aint base_address;
    MPI_Get_address(&packet_instance, &base_address);

    MPI_Get_address(&packet_instance.atom_id,   &displacements[0]);
    MPI_Get_address(&packet_instance.atom_type, &displacements[1]);
    MPI_Get_address(&packet_instance.px,        &displacements[2]);
    MPI_Get_address(&packet_instance.py,        &displacements[3]);
    MPI_Get_address(&packet_instance.pz,        &displacements[4]);
    MPI_Get_address(&packet_instance.vx,        &displacements[5]); 
    MPI_Get_address(&packet_instance.vy,        &displacements[6]); 
    MPI_Get_address(&packet_instance.vz,        &displacements[7]); 

    // 将绝对地址转换为相对位移
    for (int i = 0; i < count; ++i) {
        displacements[i] -= base_address;
    }

    // 创建并提交类型
    MPI_Type_create_struct(count, blocklengths, displacements, types, new_type);
    MPI_Type_commit(new_type);
}

/**
 * @brief 为 ChargeAtomDataPacket创建并提交 MPI 派生数据类型。
 */
void  CreateMpiChargeAtomDataPacket(MPI_Datatype* new_type) {
    // 成员数量更新为 9
    const int count = 9;

    // 块长度
    int blocklengths[count] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    // 成员的 MPI 数据类型
    MPI_Datatype types[count] = {
        MPI_RBMD_ID,   // atom_id
        MPI_RBMD_ID,   // atom_type
        MPI_RBMD_REAL, // px
        MPI_RBMD_REAL, // py
        MPI_RBMD_REAL, // pz
        MPI_RBMD_REAL, // vx
        MPI_RBMD_REAL, // vy
        MPI_RBMD_REAL, // vz
        MPI_RBMD_REAL  // charge
    };

    // 内存位移
    MPI_Aint displacements[count];

    // 获取地址和计算位移
    ChargeAtomDataPacket packet_instance;
    MPI_Aint base_address;
    MPI_Get_address(&packet_instance, &base_address);

    MPI_Get_address(&packet_instance.atom_id,   &displacements[0]);
    MPI_Get_address(&packet_instance.atom_type, &displacements[1]);
    MPI_Get_address(&packet_instance.px,        &displacements[2]);
    MPI_Get_address(&packet_instance.py,        &displacements[3]);
    MPI_Get_address(&packet_instance.pz,        &displacements[4]);
    MPI_Get_address(&packet_instance.vx,        &displacements[5]);
    MPI_Get_address(&packet_instance.vy,        &displacements[6]);
    MPI_Get_address(&packet_instance.vz,        &displacements[7]);
    MPI_Get_address(&packet_instance.charge,    &displacements[8]);

    for (int i = 0; i < count; ++i) {
        displacements[i] -= base_address;
    }

    // 创建并提交类型
    MPI_Type_create_struct(count, blocklengths, displacements, types, new_type);
    MPI_Type_commit(new_type);
}


/**
 * @brief 为 FullAtomDataPacket创建并提交 MPI 派生数据类型。
 */
void CreateMpiFullAtomDataPacket(MPI_Datatype* new_type) {
    // 成员数量更新为 13
    const int count = 13;

    // 块长度
    int blocklengths[count] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // 成员的 MPI 数据类型
    MPI_Datatype types[count] = {
        MPI_RBMD_ID,   // atom_id
        MPI_RBMD_ID,   // atom_type
        MPI_RBMD_REAL, // px
        MPI_RBMD_REAL, // py
        MPI_RBMD_REAL, // pz
        MPI_RBMD_REAL, // vx
        MPI_RBMD_REAL, // vy
        MPI_RBMD_REAL, // vz
        MPI_RBMD_REAL, // charge
        MPI_RBMD_ID,   // molecules_id
        MPI_INT,       // image_x
        MPI_INT,       // image_y
        MPI_INT        // image_z
    };

    // 内存位移
    MPI_Aint displacements[count];

    // 获取地址和计算位移
    FullAtomDataPacket packet_instance;
    MPI_Aint base_address;
    MPI_Get_address(&packet_instance, &base_address);

    MPI_Get_address(&packet_instance.atom_id,      &displacements[0]);
    MPI_Get_address(&packet_instance.atom_type,    &displacements[1]);
    MPI_Get_address(&packet_instance.px,           &displacements[2]);
    MPI_Get_address(&packet_instance.py,           &displacements[3]);
    MPI_Get_address(&packet_instance.pz,           &displacements[4]);
    MPI_Get_address(&packet_instance.vx,           &displacements[5]);
    MPI_Get_address(&packet_instance.vy,           &displacements[6]);
    MPI_Get_address(&packet_instance.vz,           &displacements[7]);
    MPI_Get_address(&packet_instance.charge,       &displacements[8]);
    MPI_Get_address(&packet_instance.molecules_id, &displacements[9]);
    MPI_Get_address(&packet_instance.image_x,      &displacements[10]);
    MPI_Get_address(&packet_instance.image_y,      &displacements[11]);
    MPI_Get_address(&packet_instance.image_z,      &displacements[12]);

    for (int i = 0; i < count; ++i) {
        displacements[i] -= base_address;
    }

    // 创建并提交类型
    MPI_Type_create_struct(count, blocklengths, displacements, types, new_type);
    MPI_Type_commit(new_type);
}
