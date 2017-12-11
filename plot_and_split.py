from scipy import cluster


def eliminate_minor_states(businesses):
    states = set(businesses['state'])
    state_num = dict()
    for state in states:
        num_in_state = businesses[businesses['state'] == state].shape[0]
        if num_in_state > 500:
#             print(state,  num_in_state)
            state_num[state] = num_in_state
    businesses = businesses.loc[businesses['state'].isin(state_num.keys())]
    return businesses, state_num


def get_user_active_weeks_in_city(users, reviews):
    """ 
    Given a set of users and a set of reviews This function determines the percentage of active weeks that a user
    has placed a review in any given city (from the set of all cities which have a review)
    """
    
    # collect all cities in the dataset
    # For each user get all reviews associated with that user
    # Divide those reviews into buckets based off of weeks and cities
    # calculate percentage of weeks in each city
    pass


def get_all_cities(businesses):
    """ Returns a list of all the cities in a dataset """
    cities = set(businesses['city'])
    return cities

def cluster_cities(businesses, k=11, iter=500, init=None):
    positions = businesses[['latitude', 'longitude']]
    if init is not None:
        clustering = cluster.vq.kmeans2(positions, init, iter=iter, minit='matrix')
    else:
        clustering = cluster.vq.kmeans2(positions, k, iter=iter, minit='points')
    return clustering

def plot_clusters_on_map(clusters):
    """ Plot the cluster of cities on a map"""

    bmap = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

    # load the shapefile, use the name 'states'
    bmap.readshapefile('st99_d00', name='states', drawbounds=True)


    # Get the location of each city and plot it
#     geolocator = Nominatim()
    for cluster in clusters:
#         print(cluster[0], cluster[1])
        x, y = bmap(cluster[1], cluster[0])
        bmap.plot(x, y,marker='o',color='Red', markersize=10) #,markersize=int(math.sqrt(count))*scale)
    plt.show()
    
def plot_all_points_US(points, clusters = None, savename=None):
    plt.figure(figsize=(10, 10))
    bmap = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

    # load the shapefile, use the name 'states'
    bmap.readshapefile('map/st99_d00', name='states', drawbounds=True)

    x, y = bmap(np.array(points['longitude']), np.array(points['latitude']))
    bmap.plot(x, y, '.', color='Red', markersize=10) #,markersize=int(math.sqrt(count))*scale)
    
    if clusters is not None:
        x, y = bmap(np.array(clusters['longitude']), np.array(clusters['latitude']))
#         print(x, y)
        bmap.plot(x, y, 'o', color='Blue', markersize=15) #,markersize=int(math.sqrt(count))*scale)
    if savename is not None:
        plt.savefig(savename)
    plt.show()


def plot_all_points_EU(points, clusters = None, savename=None):
    plt.figure(figsize=(5, 5))
    m = Basemap(llcrnrlon=-14,llcrnrlat=35,urcrnrlon=24,urcrnrlat=62,
        projection='lcc',lat_1=32,lat_2=45,lon_0=7)
    m.drawcoastlines()
    x, y = m(np.array(points['longitude']), np.array(points['latitude']))
    m.plot(x, y, '.', color='Red', markersize=10) #,markersize=int(math.sqrt(count))*scale)
    
    if clusters is not None:
        x, y = m(np.array(clusters['longitude']), np.array(clusters['latitude']))
#         print(x, y)
        m.plot(x, y, 'o', color='Blue', markersize=15) #,markersize=int(math.sqrt(count))*scale)
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_all_points_world(points, clusters = None, savename=None):
    plt.figure(figsize=(30, 30))
    m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,60.))
#     m.drawmapboundary(fill_color='aqua')
    x, y = m(np.array(points['longitude']), np.array(points['latitude']))
    m.plot(x, y, '.', color='Red', markersize=10) #,markersize=int(math.sqrt(count))*scale)
    if clusters is not None:
        x, y = m(np.array(clusters['longitude']), np.array(clusters['latitude']))
#         print(x, y)
        m.plot(x, y, 'o', color='Blue', markersize=15) #,markersize=int(math.sqrt(count))*scale)
    if savename is not None:
        plt.savefig(savename)
    plt.show()


    