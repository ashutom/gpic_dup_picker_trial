from googleApisTrial import create_service
from googleApisTrial import file_validity_checker



root_path="C:\\Users\\ashmishr\\Documents\\projects\\vs_projs\\Python_trial_app\\duplicate_picker\\configs\\"
client_secret_file='configs\\Desktop_client_auth_2dot0_key.json'

API_NAME='photoslibrary'
API_VERSION='v1'
#SCOPES=['https://www.googleapis.com/auth/photoslibrary',
#        'https://www.googleapis.com/auth/photoslibrary.readonly']
SCOPES=['https://www.googleapis.com/auth/photoslibrary']


def main():
    if  not file_validity_checker(client_secret_file):
        return
    
    service = create_service(client_secret_file, API_NAME,API_VERSION,SCOPES)
    if service == None:
        print("error in service creation. Check logs")
        return
    
    albums_response = service.albums().list().execute()
    albums = albums_response['albums']

    nexPageToken = albums_response.get('nexPageToken')

    while nexPageToken:
        print('fetching next page ...')
        albums_response=service.albums().list(pageToken=nexPageToken).execute()
        albums.extend(albums_response['albums'])
        nexPageToken = albums_response.get('nexPageToken')

    for album in albums:
        count=album.get('mediaitems',0)
        print(f'ID = {album['id']}')
        print(f'title = {album['title']}')

        print(f'count = {count}')
        print(f'End ...  \n\n')
        
    #mediaitems_response = service.mediaitems().list().execute()
    #pics = mediaitems_response['mediaitems']
    #for pic in pics:
    #    mediaMetadata=pic['mediaMetadata']



if __name__ == '__main__':
    main()
