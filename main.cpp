#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>//体素下采样滤波
#include <pcl/features/normal_3d_omp.h>//使用OMP需要添加的头文件
#include <pcl/features/fpfh_omp.h> //fpfh加速计算的omp(多核并行计算)
#include <pcl/registration/ia_ransac.h>//sac_ia算法
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <cmath>
#include <ctime>
#include "myicp.h"
#include "myicp_helpers.h"

using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;//快速点特征直方图（提取特征描述子）

fpfhFeature::Ptr compute_fpfh_feature(pointcloud::Ptr input_cloud, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree)
{
    //-------------------------法向量估计-----------------------
    pointnormal::Ptr normals(new pointnormal);
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//模板类实例化 n
    n.setInputCloud(input_cloud);
    n.setNumberOfThreads(8);//设置openMP的线程数
    n.setSearchMethod(tree);
    n.setRadiusSearch(2);
    n.compute(*normals);
    //------------------FPFH估计-------------------------------
    fpfhFeature::Ptr fpfh(new fpfhFeature);
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> f;
    f.setNumberOfThreads(8); //指定8核计算
    f.setInputCloud(input_cloud);
    f.setInputNormals(normals);
    f.setSearchMethod(tree);
    f.setRadiusSearch(4);
    f.compute(*fpfh);

    return fpfh;

}

void visualize_pcd(pointcloud::Ptr pcd_src, pointcloud::Ptr pcd_tgt, pointcloud::Ptr pcd_final)
{

    pcl::visualization::PCLVisualizer viewer("registration Viewer");
    //--------创建两个显示窗口并设置背景颜色------------
    int v1, v2;
    viewer.createViewPort(0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer.setBackgroundColor(0, 0, 0, v1);
    viewer.setBackgroundColor(0.05, 0, 0, v2);
    //-----------给点云添加颜色-------------------------
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(pcd_src, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(pcd_tgt, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h(pcd_final, 0, 255, 0);
    //----------添加点云到显示窗口----------------------
    viewer.addPointCloud(pcd_src, src_h, "source cloud", v1);
    viewer.addPointCloud(pcd_tgt, tgt_h, "target cloud", v1);
    viewer.addPointCloud(pcd_tgt, tgt_h, "tgt cloud", v2);
    viewer.addPointCloud(pcd_final, final_h, "final cloud", v2);

    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

PointCloud::Ptr scaling(PointCloud::Ptr cloud_src_o, PointCloud::Ptr cloud_tgt_o) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr s_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    double X1_o = 0, Y1_o = 0, Z1_o = 0, X2_o = 0, Y2_o = 0, Z2_o = 0;
    double X_1 = 0, Y_1 = 0, Z_1 = 0, X_2 = 0, Y_2 = 0, Z_2 = 0;
    double scale_x, scale_y, scale_z, scale;
    for (size_t i = 0; i < cloud_src_o->points.size(); ++i)
    {
        X1_o += cloud_src_o->points[i].x;
        Y1_o += cloud_src_o->points[i].y;
        Z1_o += cloud_src_o->points[i].z;
    }

    for (size_t i = 0; i < cloud_tgt_o->points.size(); ++i)
    {
        X2_o += cloud_tgt_o->points[i].x;
        Y2_o += cloud_tgt_o->points[i].y;
        Z2_o += cloud_tgt_o->points[i].z;
    }

    X1_o /= cloud_src_o->points.size();
    Y1_o /= cloud_src_o->points.size();
    Z1_o /= cloud_src_o->points.size();

    X2_o /= cloud_tgt_o->points.size();
    Y2_o /= cloud_tgt_o->points.size();
    Z2_o /= cloud_tgt_o->points.size();//求平均值

    for (size_t i = 0; i < cloud_src_o->points.size(); ++i)
    {
        X_1 += sqrt(pow(cloud_src_o->points[i].x - X1_o, 2));
        Y_1 += sqrt(pow(cloud_src_o->points[i].y - Y1_o, 2));
        Z_1 += sqrt(pow(cloud_src_o->points[i].z - Z1_o, 2));
    }

    for (size_t i = 0; i < cloud_tgt_o->points.size(); ++i)
    {
        X_2 += sqrt(pow(cloud_tgt_o->points[i].x - X2_o, 2));
        Y_2 += sqrt(pow(cloud_tgt_o->points[i].y - Y2_o, 2));
        Z_2 += sqrt(pow(cloud_tgt_o->points[i].z - Z2_o, 2));
    }
    X_1 /= cloud_src_o->points.size();
    Y_1 /= cloud_src_o->points.size();
    Z_1 /= cloud_src_o->points.size();

    X_2 /= cloud_tgt_o->points.size();
    Y_2 /= cloud_tgt_o->points.size();
    Z_2 /= cloud_tgt_o->points.size();//方差

    pcl::PointXYZ minPt_1, maxPt_1, minPt_2, maxPt_2;
    scale_x = sqrt(pow(X_2, 2)) / sqrt(pow(X_1, 2));
    scale_y = sqrt(pow(Y_2, 2)) / sqrt(pow(Y_1, 2));
    scale_z = sqrt(pow(Z_2, 2)) / sqrt(pow(Z_1, 2));
    scale = sqrt(pow(X_2, 2) + pow(Y_2, 2) + pow(Z_2, 2)) / sqrt(pow(X_1, 2) + pow(Y_1, 2) + pow(Z_1, 2));
    // 缩放
    std::cout << scale << endl;
    Eigen::Isometry3d S = Eigen::Isometry3d::Identity();
    S = Eigen::Scaling(scale, scale, scale);
    // 执行缩放变换，并将结果保存在 s_cloud 中
    pcl::transformPointCloud(*cloud_src_o, *s_cloud, S.matrix());
    return s_cloud;
}


int main(int argc, char** argv)
{

    clock_t start, end, time;
    start = clock();
    pointcloud::Ptr source_cloud(new pointcloud);
    pointcloud::Ptr target_cloud0(new pointcloud);

    pcl::io::loadPCDFile<pcl::PointXYZ>("./data/crown_lower_seg.pcd", *source_cloud);
    pcl::io::loadPCDFile<pcl::PointXYZ>("./data/lowTeeth_2bigger.pcd", *target_cloud0);

    //-------------------------尺度初始统一-------------------------
    PointCloud::Ptr target_cloud(new PointCloud);
    target_cloud = scaling(target_cloud0, source_cloud);
    visualize_pcd(target_cloud0, source_cloud,  target_cloud);

    //-------------------------源点云下采样滤波-------------------------
    pcl::VoxelGrid<pcl::PointXYZ> vs;
    vs.setLeafSize(0.8, 0.8, 0.8);//0.8,76482->5650
    vs.setInputCloud(source_cloud);
    pointcloud::Ptr source(new pointcloud);
    vs.filter(*source);
    cout << "down size *source_cloud from " << source_cloud->size() << " to " << source->size() << endl;

    //----------------------目标点云下采样滤波-------------------------
    pcl::VoxelGrid<pcl::PointXYZ> vt;
    vt.setLeafSize(0.8, 0.8, 0.8);//1.2，161286->6398
    vt.setInputCloud(target_cloud);
    pointcloud::Ptr target(new pointcloud);
    vt.filter(*target);
    cout << "down size *target_cloud from " << target_cloud->size() << " to " << target->size() << endl;
    //---------------计算源点云和目标点云的FPFH------------------------
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source, tree);
    fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target, tree);

    //--------------采样一致性SAC_IA初始配准----------------------------
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(source);
    sac_ia.setSourceFeatures(source_fpfh);
    sac_ia.setInputTarget(target);
    sac_ia.setTargetFeatures(target_fpfh);
    sac_ia.setMinSampleDistance(0.1);//设置样本之间的最小距离
    sac_ia.setCorrespondenceRandomness(6); //在选择随机特征对应时，设置要使用的邻居的数量;

    pointcloud::Ptr align(new pointcloud);
    sac_ia.align(*align);//对齐
    end = clock();
    Eigen::Matrix4f sac_trans;
    sac_trans = sac_ia.getFinalTransformation();
    //pcl::transformPointCloud(*source, *align, sac_ia.getFinalTransformation());
    // pcl::io::savePCDFile("crou_output.pcd", *align);
    cout << "calculate time is: " << float(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "\nSAC_IA has converged, score is " << sac_ia.getFitnessScore() << endl;
    cout << "变换矩阵：\n" << sac_ia.getFinalTransformation() << endl;

    PointCloud::Ptr sac_result(new PointCloud);
    pcl::transformPointCloud(*source_cloud, *sac_result, sac_trans);
    //pcl::io::savePCDFileASCII("..\\output\\scale_sac.pcd", *sac_result);
    visualize_pcd(source_cloud, target_cloud, sac_result);
    

    //------------非scaleICP-------------
/*
    PointCloud::Ptr icp_result(new PointCloud);
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(3);
    icp.setMaximumIterations(5000);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(0.001);//误差平方和小于该阈值，即找到的解
    icp.align(*icp_result, sac_trans);//这里的icp变换矩阵的初始矩阵是sac变换矩阵，不是单位阵

    std::cout << "ICP has converged:" << icp.hasConverged()
        << " score: " << icp.getFitnessScore() << std::endl;
    Eigen::Matrix4f icp_trans;
    icp_trans = icp.getFinalTransformation();
    std::cout << icp_trans << endl;
    pcl::transformPointCloud(*source, *icp_result, icp_trans);

    PointCloud::Ptr icp_result2(new PointCloud);
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp2;
    icp2.setInputSource(source);
    icp2.setInputTarget(target);
    icp2.setMaxCorrespondenceDistance(3);
    icp2.setMaximumIterations(5000);
    icp2.setTransformationEpsilon(1e-10);
    icp2.setEuclideanFitnessEpsilon(0.001);//误差平方和小于该阈值，即找到的解
    icp2.align(*icp_result2, icp_trans);

    std::cout << "ICP has converged:" << icp2.hasConverged()
        << " score: " << icp2.getFitnessScore() << std::endl;
    Eigen::Matrix4f icp_trans2;
    icp_trans2 = icp2.getFinalTransformation();
    std::cout << icp_trans2 << endl;
    //使用创建的变换对未过滤的输入点云进行变换
    pcl::transformPointCloud(*source_cloud, *icp_result2, icp_trans2);

    //保存转换的输入点云
    pcl::io::savePCDFileASCII("..\\output\\scale_crown_aligned.pcd", *icp_result2);

    //-------------------可视化------------------------------------
    visualize_pcd(source_cloud, target_cloud, icp_result2);
*/

    //----------------scaleICP---------------
    MyICP icp;
    icp.setSourceCloud(sac_result);
    icp.setTargetCloud(target_cloud);
    icp.setLeafSize(0.8);
    icp.downsample();
    icp.setMinError(0.05);
    icp.setMaxIters(100);
    icp.registration();
    icp.saveICPCloud("./output/ScaleICP.pcd");
    icp.getTransformationMatrix();
    icp.getScore();
    icp.visualize();

    return 0;
}
